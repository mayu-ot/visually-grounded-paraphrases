import numpy as np
import chainer
from chainer.iterators import SerialIterator
import pandas as pd
import imageio
from chainer.dataset.convert import to_device
import sys
sys.path.append('./')
from func.datasets.converters import jitter_bbox, cvrt_bbox
from func.datasets.datasets import BBoxDataset
from func.nets.faster_rcnn import FasterRCNNExtractor

def save_jitter_frcnn_feat(split, device):
    model = FasterRCNNExtractor(n_fg_class=20, pretrained_model='voc07')
    # model = iParaphraseNet()
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    data = BBoxDataset(split)
    data_itr = SerialIterator(data, batch_size=100, repeat=False, shuffle=False)

    feat = np.zeros((len(data), 4096)).astype('f')
    roi_arr = np.zeros((len(data), 4)).astype('f')
    roi_indices = model.xp.zeros((1,)).astype('i')
    b_size = 30

    for i in range(len(data)):
        batch = [data[i]]
        im, roi = cvrt_bbox(batch, device, aspect_band=2/3, offset_band=0.4)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            im = im[0]
            y = model.extract(im[None, :], roi, roi_indices)

        y.to_cpu()
        feat[i,:] = y.data[:]
        roi = chainer.cuda.to_cpu(roi)
        roi_arr[i, :] = roi

    np.save('data/region_feat/jitter_roi-frcnn_asp0.66_off0.4/%s'%split, feat)
    np.save('data/region_feat/jitter_roi-frcnn_asp0.66_off0.4/rois_%s'%split, roi_arr)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-s', type=str, default='val')
    parser.add_argument('--device', '-d', type=int, default=0)
    args = parser.parse_args()
    
    save_jitter_frcnn_feat(args.split, args.device)