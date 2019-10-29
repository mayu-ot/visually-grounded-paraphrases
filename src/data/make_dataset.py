# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from gensim.models import KeyedVectors
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import (
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
    strip_numeric,
)
import numpy as np


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # prepare_word_embedding()

    get_bbox_file("data/raw/ddpn_train.csv", "data/interim/ddpn_bbox_train.csv")
    get_bbox_file("data/raw/ddpn_val.csv", "data/interim/ddpn_bbox_val.csv")
    get_bbox_file("data/raw/ddpn_test.csv", "data/interim/ddpn_bbox_test.csv")

    get_bbox_file("data/raw/plclc_train.csv", "data/interim/plclc_bbox_train.csv")
    get_bbox_file("data/raw/plclc_val.csv", "data/interim/plclc_bbox_val.csv")
    get_bbox_file("data/raw/plclc_test.csv", "data/interim/plclc_bbox_test.csv")


def prepare_word_embedding():
    """Construct vocabulary file and word embedding file.
    """
    df = pd.read_csv(
        "data/raw/train.csv", usecols=["original_phrase1", "original_phrase2", "ytrue"]
    )

    model = KeyedVectors.load_word2vec_format(
        "/data/mayu-ot/Data/Model/GoogleNews-vectors-negative300.bin.gz", binary=True
    )

    CUSTOM_FILTERS = [
        lambda x: x.lower(),
        strip_punctuation,
        strip_multiple_whitespaces,
        strip_numeric,
    ]

    doc = [preprocess_string(x, CUSTOM_FILTERS) for x in df.values[:, :2].ravel()]

    dct = Dictionary(doc)

    bad_ids = []
    for k, v in dct.iteritems():
        if v not in model:
            bad_ids.append(k)
    dct.filter_tokens(bad_ids)

    dct.compactify()

    for k, v in dct.iteritems():
        print(k, v)
        if k == 10:
            break

    dct.save_as_text("data/processed/dictionary.txt")

    word_emb = np.ones((len(dct), 300))

    for k, v in dct.iteritems():
        word_emb[k] = model[v]

    np.save("data/processed/word2vec", word_emb)


def get_bbox_file(file_name, out_name):
    """Assign ids for detected bounding boxes.
    
    Arguments:
        file_name {str} -- input file path. The file contains bounding boxes and corresponding phrases.
        out_name {str} -- output file path. The file has bounding boxes corresponding ids. Duplicates will be removed.
    """
    logging.info("Assign ids for detected bounding boxes.")
    logging.info("loading: %s" % file_name)

    df = pd.read_csv(file_name)
    df = df[["xmin", "ymin", "xmax", "ymax", "image"]]
    df = df.drop_duplicates()
    df = df.reset_index()

    logging.info("writing: %s" % out_name)
    df.to_csv(out_name)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
