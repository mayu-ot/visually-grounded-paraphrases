import os
import h5py
import numpy as np
import progressbar

split = 'train'
in_dir = '/mnt/fs1/chu/experiments/iparaphrasing/pl-clc/paraphrase/region_feature.%sSplit.cca/' % split
out_file = 'data/region_feat/cca/%s.h5' % split

FILES = os.listdir(in_dir)

bar = progressbar.ProgressBar()

with h5py.File(out_file, 'w') as hf:
    for fname in bar(FILES):
        x = np.loadtxt(in_dir + fname).astype(np.float32)
        hf.create_dataset(fname, data=x)
