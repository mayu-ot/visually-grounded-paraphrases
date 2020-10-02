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
            phrase_idx = dictionary.doc2idx(phrase, None)
            phrase_idx = [x for x in phrase_idx if x is not None]
            numerized_phrases.append(phrase_idx)

        return numerized_phrases

    def get_example(self, i: int) -> Tuple[List[int], List[int]]:
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


def bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> np.ndarray:
    tl = np.maximum(bbox_a[:, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    iou = area_i / (area_a + area_b - area_i)
    return iou


@dataclass
class IoUDataLoader(AbstractDataLoader):
    feat_type: str

    def __post_init__(self) -> None:
        pairs: pd.DataFrame = self.load_dataset_file(self.dataset_file)

        self.visfeat_idx_a: List[int] = pairs["visfeat_idx_a"].tolist()
        self.visfeat_idx_b: List[int] = pairs["visfeat_idx_b"].tolist()

        ddpn = pd.read_csv(
            f"data/raw/{self.feat_type}_{self.split}.csv",
            usecols=["xmin", "ymin", "xmax", "ymax"],
        )
        bbox = ddpn.values

        bbox_a = bbox[pairs.visfeat_idx_a]
        bbox_b = bbox[pairs.visfeat_idx_b]
        ious = bbox_iou(bbox_a, bbox_b)
        self.ious = ious.tolist()

    def __len__(self) -> int:
        return len(self.ious)

    def get_example(self, i: int) -> float:
        return self.ious[i]


@dataclass
class PhraseOnlyDataLoader(AbstractDataLoader):
    def __post_init__(self) -> None:
        pairs: pd.DataFrame = self.load_dataset_file(self.dataset_file)
        self.image = pairs["image"].tolist()
        self.label = pairs["label"].tolist()

        self.phrase_dataloader = PhraseDataLoader(
            self.dataset_file, self.split, self.subset_size
        )

    def __len__(self) -> int:
        return len(self.label)

    def get_example(
        self, i
    ) -> Tuple[List[int], List[int], np.ndarray, np.ndarray, bool]:
        phr_a, phr_b = self.phrase_dataloader[i]
        label = self.label[i]
        return phr_a, phr_b, label


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

    def get_example(
        self, i
    ) -> Tuple[List[int], List[int], np.ndarray, np.ndarray, bool]:
        phr_a, phr_b = self.phrase_dataloader[i]
        vis_a, vis_b = self.visual_dataloader[i]
        label = self.label[i]
        return phr_a, phr_b, vis_a, vis_b, label


@dataclass
class MultiModalIoUDataLoader(MultiModalDataLoader):
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

        self.iou_dataloader = IoUDataLoader(
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
        iou = self.iou_dataloader[i]
        label = self.label[i]
        return phr_a, phr_b, vis_a, vis_b, iou, label


@dataclass()
class DownSampler(object):
    indices_positive: List[int]
    indices_negative: List[int]

    def __call__(self, current_order, current_position):
        n_positives = len(self.indices_positive)
        n_drops = len(self.indices_negative) - n_positives
        n_drops = n_drops // 2
        n_samples = len(self.indices_negative) - n_drops
        selected_indices: List[int] = random.sample(
            self.indices_negative, n_samples
        )
        selected_indices += self.indices_positive
        return np.random.permutation(selected_indices)
