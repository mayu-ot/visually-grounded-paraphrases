import numpy as np
import pandas as pd
from chainer.dataset import DatasetMixin
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import (
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
    strip_numeric,
)
from chainer.dataset.convert import to_device

def preprocess_phrase(texts):
    CUSTOM_FILTERS = [
        lambda x: x.lower(),
        strip_punctuation,
        strip_multiple_whitespaces,
        strip_numeric,
    ]
    processed = [preprocess_string(x, CUSTOM_FILTERS) for x in texts]
    return processed

class Dataset(DatasetMixin):
    def __init__(self, method, split, san_check=False):
        names = ['image', 'phrase1', 'phrase2', 'label', 'v_idx1', 'v_idx2']
        df = pd.read_csv(f"data/processed/{method}/{split}.csv",
                         names=names,
                         sep=',',
                         header=0)

        if san_check:
            df = df[:500]
        
        text1 = preprocess_phrase(df['phrase1'])
        text2 = preprocess_phrase(df['phrase2'])
        
        self.df = df
        self.data = [x for x in zip(text1, text2, df['v_idx1'], df['v_idx2'], df['label'])]
        
        
        self.vis_feat = np.load(f"data/processed/{method}/feat_{split}.npy").astype('f')
        self.dct = Dictionary.load_from_text('data/processed/dictionary.txt')
        if split=='train':
            self.downsample()
        else:
            self.index = df.index.to_list()
            
    def downsample(self):
        df = self.df
        pos_index = df[df['label']].index.to_list()
        neg_index = df[~df['label']].index.to_list()
        neg_index = np.random.permutation(neg_index)[:len(pos_index)]
        neg_index = neg_index.tolist()
        index = pos_index + neg_index
        index.sort()
        self.index = index
        
    def __len__(self):
        return len(self.index)
    
    def get_example(self, i):
        ind = self.index[i]
        p1, p2, v_id1, v_id2, l = self.data[ind]
        
        p1 = self.dct.doc2idx(p1)
        p2 = self.dct.doc2idx(p2)
        
        if len(p1)==0:
            p1 = [-1]
        
        if len(p2)==0:
            p2 = [-1]
        
        v1 = self.vis_feat[v_id1]
        v2 = self.vis_feat[v_id2]
        
        return p1, p2, v1, v2, l, 

def conv_f(batch, device=None):
    p1 = [to_device(device, np.asarray(b[0]).astype('i')) for b in batch]
    p2 = [to_device(device, np.asarray(b[1]).astype('i')) for b in batch]
    v1 = [b[2] for b in batch]
    v2 = [b[3] for b in batch]
    l = [b[4] for b in batch]
    
    v1 = np.vstack(v1).astype('f')
    v2 = np.vstack(v2).astype('f')
    l = np.asarray(l).astype('i')
    
    v1 = to_device(device, v1)
    v2 = to_device(device, v2)
    l = to_device(device, l)
    
    return p1, p2, v1, v2, l
    