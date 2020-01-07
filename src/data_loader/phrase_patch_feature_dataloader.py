import numpy as np
from chainer.dataset import DatasetMixin
import pandas as pd
from logging import getLogger
import logging

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

class Loader(DatasetMixin):
    def __init__(self, split, san_check):
        df = pd.read_csv(
            "data/raw/%s.csv" % split,
            usecols=["image", "original_phrase1", "original_phrase2", "ytrue"],
        )

        if san_check:
            df = df.iloc[np.random.permutation(len(df))[:1000]]

        # preprocess text
        logger.info("preprocessing text")
        