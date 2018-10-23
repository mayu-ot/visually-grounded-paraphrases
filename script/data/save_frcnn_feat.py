import numpy as np
import chainer
import sys
sys.path.append('./')
from func.datasets.converters import cvrt_frcnn_input
from func.datasets.datasets import DDPNBBoxDataset, PLCLCBBoxDataset
from func.nets.faster_rcnn import FasterRCNNExtractor
import progressbar

def extract_frcnn_feat(bbox_data, split, device):
    model = FasterRCNNExtractor(n_fg_class=20, pretrained_model='voc07')
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    feat = np.zeros((len(bbox_data), 4096)).astype('f')
    roi_indices = model.xp.zeros((1,)).astype('i')
    b_size = 30

    for i in progressbar.progressbar(range(len(bbox_data))):
        batch = [bbox_data[i]]
        im, roi = cvrt_frcnn_input(batch, device)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            im = im[0]
            y = model.extract(im[None, :], roi, roi_indices)

        y.to_cpu()
        feat[i,:] = y.data[:]
        
    return feat
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', '-s', type=str, default='val')
    parser.add_argument('--device', '-d', type=int, default=0)
    parser.add_argument('--method', type=str)
    args = parser.parse_args()
    
    if args.method == 'ddpn':
        bbox_data = DDPNBBoxDataset(args.split)
    elif args.method == 'plclc':
        bbox_data = PLCLCBBoxDataset(args.split)
        
    feat = extract_frcnn_feat(bbox_data, args.split, args.device)
    np.save('data/region_feat/%s_roi-frcnn/%s'%(args.method, args.split), feat)