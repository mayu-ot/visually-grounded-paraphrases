import pandas as pd
import os
import sys
sys.path.append('func/nets/')
from faster_rcnn import FasterRCNNExtractor
from chainercv.datasets import voc_bbox_label_names
from chainer import cuda
from chainercv import utils
import numpy as np
import chainer

def load_roi(roi):
    roi = roi[1:-1]
    roi = [int(x) for x in roi.split(', ')]
    return roi

def main(split, img_root, device=0):     
    
    model = FasterRCNNExtractor(n_fg_class=len(voc_bbox_label_names), pretrained_model='voc07')
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()  # Make the GPU current
        model.to_gpu()
        
    in_file = 'data/gt-roi/phrase_pair_%s.csv'%split
    df = pd.read_csv(in_file)
    
    for d_i in range(2):
        data = {}
        images = df.image.unique()

        for im in images:
            bbox = df.query('image == %i' % im)[['roi%i' % (d_i+1)]].values
            data[im] = np.asarray([load_roi(x) for x in bbox.ravel()]).astype(np.float32)


        N = len(df)
        feat = np.zeros((N, 4096), dtype=np.float32)
        print('extract features of %i regions'%N)

        j = 0

        for im in images:
            x = utils.read_image(os.path.join(img_root, '%i.jpg'%im), color=True)
            bbox = data[im]
            roi_indices = np.zeros((len(bbox),), dtype=np.int32)

            # preprocess
            p_x = model.prepare(x)
            scale = p_x.shape[-1] * 1. / x.shape[-1]
            bbox = bbox * scale

            # to gpu
            p_x = cuda.to_gpu(p_x)
            bbox = cuda.to_gpu(bbox)
            roi_indices = cuda.to_gpu(roi_indices)
            with chainer.using_config('train', False):
                y = model.extract(p_x[None, :], bbox, roi_indices)
            y.to_cpu()
            y_arr = y.data

            feat[j:j+len(y_arr)] = y_arr
            j += len(y_arr)

            print('%12s: %i / %i' % (im, j, N))

        np.save('data/region_feat/gt-roi-frcnn/%i-%s'%(d_i+1, split), feat)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-s', type=str, default='val')
    parser.add_argument('--img_root', '-i', type=str, default='./')
    parser.add_argument('--device', '-d', type=int, default=0)
    args = parser.parse_args()
    main(args.split, args.img_root, args.device)