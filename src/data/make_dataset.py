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
from tqdm import tqdm
from extract_visual_feature import extract_frcnn_feat


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
#     build_dataset()
    extract_visual_feature()

def search(df_v, p1, p2):
    idx1 = df_v.index[df_v["phrase"] == p1][0]
    idx2 = df_v.index[df_v["phrase"] == p2][0]
    return idx1, idx2
    
def build_dataset():

    for split in ['train']:

        bbox_df = pd.read_csv(f'data/raw/ddpn_{split}.csv')
        pair_df = pd.read_csv(f'data/raw/{split}.csv')
        bbox_df = bbox_df[['image', 'phrase']]
        pair_df = pair_df[['image', 'original_phrase1', 'original_phrase2', 'ytrue']]

        out_df = []

        all_imgs = pair_df["image"].unique()
        for im_id in tqdm(all_imgs):
            df_q = pair_df[pair_df["image"] == im_id].copy()
            df_v = bbox_df[bbox_df["image"] == im_id]

            args = [(df_v, p1, p2) for _, p1, p2, _ in df_q.itertuples(index=False)]
            
            res = [search(*x) for x in args]

            df_q['visfeat_idx1'] = [x for x, _ in res]
            df_q['visfeat_idx2'] = [x for _, x in res]

            out_df.append(df_q)

        out_df = pd.concat(out_df)
        out_df.to_csv(f'data/processed/ddpn/{split}.csv')

def extract_visual_feature():
    for split in ['val', 'test', 'train']:
        feat = extract_frcnn_feat(f'data/raw/ddpn_{split}.csv', 0)
        np.save(f"data/processed/ddpn/feat_{split}", feat)

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
