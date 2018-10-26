import os
import h5py
import numpy as np
import progressbar

for split in ['train', 'test', 'val']:
    in_dir = 'data/org_files/region_id.%sSplit/' % split
    out_file = 'data/region_feat/roi/full_%s.h5' % split

    FILES = os.listdir(in_dir)

    bar = progressbar.ProgressBar()

    with h5py.File(out_file, 'w') as hf:
        for fname in bar(FILES):
            hf.create_dataset(
                fname, data=np.loadtxt(in_dir + fname).astype(np.uint32))
