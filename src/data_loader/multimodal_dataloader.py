import numpy as np
import pandas as pd
from chainer.dataset import DatasetMixin
from dataclasses import dataclass
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
)
from gensim.corpora.dictionary import Dictionary
from typing import List, Iterable, Callable, Tuple
import random


@dataclass
class AbstractDataLoader(DatasetMixin):
    dataset_file: str
    split: str
    subset_size: float

    def load_dataset_file(self, dataset_file: str) -> pd.DataFrame:

        dataset = pd.read_csv(
            self.dataset_file,
            names=[
                "image",
                "phrase_a",
                "phrase_b",
                "label",
                "visfeat_idx_a",
                "visfeat_idx_b",
            ],
            header=0,
        )

        if self.subset_size < 1.0:
            subset_num = int(len(dataset) * self.subset_size)
            dataset = dataset.sample(subset_num, random_state=1234)

        return dataset

    @property
    def indices_positive(self) -> List[int]:
        return [i for i, l in enumerate(self.label) if l]

    @property
    def indices_negative(self) -> List[int]:
        return [i for i, l in enumerate(self.label) if not l]


@dataclass
class PhraseDataLoader(AbstractDataLoader):
    def __post_init__(self) -> None:
        pairs: pd.DataFrame = self.load_dataset_file(self.dataset_file)

        dct = Dictionary.load_from_text(("data/processed/dictionary.txt"))

        self.phrase_a = self.preprocess_phrase(pairs["phrase_a"], dct)
        self.phrase_b = self.preprocess_phrase(pairs["phrase_b"], dct)

    def __len__(self) -> int:
        return len(self.phrase_a)

    @property
    def custom_filter(self) -> List[Callable]:
        return [
            lambda x: x.lower(),
            strip_punctuation,
            strip_multiple_whitespaces,
            strip_numeric,
        ]

    def preprocess_phrase(
        self, phrases: Iterable[str], dictionary: Dictionary
    ) -> List[List[int]]:
        numerized_phrases: List[List[int]] = []

        for phrase in phrases:
            phrase = preprocess_string(phrase, self.custom_filter)
            phrase = dictionary.doc2idx(phrase, None)
            phrase = [x for x in phrase if x is not None]
            numerized_phrases.append(phrase)

        return numerized_phrases

    def get_example(self, i):
        phr_a = self.phrase_a[i]
        phr_b = self.phrase_b[i]

        if len(phr_a) == 0:
            phr_a = [-1]

        if len(phr_b) == 0:
            phr_b = [-1]

        return phr_a, phr_b


@dataclass
class VisualDataLoader(AbstractDataLoader):
    feat_type: str

    def __post_init__(self) -> None:
        pairs: pd.DataFrame = self.load_dataset_file(self.dataset_file)
        self.visfeat_idx_a: List[int] = pairs["visfeat_idx_a"].tolist()
        self.visfeat_idx_b: List[int] = pairs["visfeat_idx_b"].tolist()

        self.vis_feat = self.load_visual_feat(self.split)

    def __len__(self) -> int:
        return len(self.visfeat_idx_a)

    def load_visual_feat(self, split: str) -> np.ndarray:
        return np.load(f"data/processed/{self.feat_type}/feat_{split}.npy")

    def get_example(self, i) -> Tuple[np.ndarray, np.ndarray]:
        vis_a = self.vis_feat[self.visfeat_idx_a[i]]
        vis_b = self.vis_feat[self.visfeat_idx_b[i]]
        return vis_a, vis_b


@dataclass
class MultiModalDataLoader(AbstractDataLoader):
    feat_type: str

    def __post_init__(self) -> None:
        pairs: pd.DataFrame = self.load_dataset_file(self.dataset_file)
        self.image = pairs["image"].tolist()
        self.label = pairs["label"].tolist()

        self.phrase_dataloader = PhraseDataLoader(
            self.dataset_file, self.split, self.subset_size
        )
        self.visual_dataloader = VisualDataLoader(
            self.dataset_file,
            self.split,
            self.subset_size,
            feat_type=self.feat_type,
        )

    def __len__(self) -> int:
        return len(self.label)

    def get_example(self, i):
        phr_a, phr_b = self.phrase_dataloader[i]
        vis_a, vis_b = self.visual_dataloader[i]
        label = self.label[i]
        return phr_a, phr_b, vis_a, vis_b, label


@dataclass()
class DownSampler(object):
    indices_positive: List[int]
    indices_negative: List[int]

    def __call__(self, current_order, current_position):
        n_positives = len(self.indices_positive)
        selected_indices: List[int] = random.sample(
            self.indices_negative, n_positives * 3
        )
        selected_indices += self.indices_positive
        return np.random.permutation(selected_indices)
