import numpy as np
import chainer
import sys
sys.path.append('./')
from func.datasets.converters import cvrt_frcnn_input
from func.datasets.datasets import PLCLCBBoxDataset
from func.nets.faster_rcnn import FasterRCNNExtractor
import progressbar

def save_plclc_frcnn_feat(split, device):
    model = FasterRCNNExtractor(n_fg_class=20, pretrained_model='voc07')
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    data = PLCLCBBoxDataset(split)

    feat = np.zeros((len(data), 4096)).astype('f')
    roi_indices = model.xp.zeros((1,)).astype('i')
    b_size = 30

    for i in progressbar.progressbar(range(len(data))):
        batch = [data[i]]
        im, roi = cvrt_frcnn_input(batch, device)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            im = im[0]
            y = model.extract(im[None, :], roi, roi_indices)

        y.to_cpu()
        feat[i,:] = y.data[:]

    np.save('data/region_feat/plclc_roi-frcnn/%s'%split, feat)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-s', type=str, default='val')
    parser.add_argument('--device', '-d', type=int, default=0)
    args = parser.parse_args()
    
    save_plclc_frcnn_feat(args.split, args.device)