import numpy as np
from chainer.dataset import DatasetMixin
import pandas as pd
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
    strip_numeric,
)
from logging import getLogger
import logging

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

np.random.seed(0)


class PhraseOnlyDataLoader(DatasetMixin):
    def __init__(self, split, san_check=False):
        # load dataset
        df = pd.read_csv(
            "data/raw/%s.csv" % split,
            usecols=["image", "original_phrase1", "original_phrase2", "ytrue"],
        )

        if san_check:
            df = df.iloc[np.random.permutation(len(df))[:1000]]

        # preprocess text
        logger.info("preprocessing text")
        CUSTOM_FILTERS = [
            lambda x: x.lower(),
            strip_punctuation,
            strip_multiple_whitespaces,
            strip_numeric,
        ]
        phrase_a = [preprocess_string(x, CUSTOM_FILTERS) for x in df.values[:, 1]]
        phrase_b = [preprocess_string(x, CUSTOM_FILTERS) for x in df.values[:, 2]]
        self.phrase_a = phrase_a
        self.phrase_b = phrase_b

        # load dictionary
        logger.info("loading a dictionary")
        self.dct = Dictionary.load_from_text(("data/processed/dictionary.txt"))

        # label
        self.label = df.values[:, 3]

        # image id
        self.image = df.values[:, 0]

        logger.info("done.")

    def __len__(self):
        return len(self.label)

    def get_example(self, i):

        phr_a = self.phrase_a[i]
        phr_b = self.phrase_b[i]

        phr_a = [x for x in self.dct.doc2idx(phr_a, None) if x is not None]
        phr_b = [x for x in self.dct.doc2idx(phr_b, None) if x is not None]

        if len(phr_a) == 0:
            phr_a = [-1]

        if len(phr_b) == 0:
            phr_b = [-1]

        label = self.label[i]

        return phr_a, phr_b, label
