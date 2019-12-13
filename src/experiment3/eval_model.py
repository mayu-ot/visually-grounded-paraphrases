import chainer
from chainer import training
from chainer.training import extensions
import os
from chainer.iterators import SerialIterator
from chainer.serializers import load_npz
import json
from datetime import datetime as dt
import sys
sys.path.append('./')

import numpy as np
import pandas as pd

from dataset import Dataset, conv_f
from snn import VGPNet
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

# chainer.config.multiproc = True  # single proc is faster

def eval(model_dir,
         split,
         b_size=500,
         device=0,
         ):
    
    test = Dataset(split)
    test_iter = SerialIterator(test, b_size, shuffle=False, repeat=False)
    
    emb_W = np.load('data/processed/word2vec.npy').astype('f')
    model = VGPNet(128, emb_W)
    load_npz(f'{model_dir}/model', model)

    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()
    
    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False
    
    pred = []
    
    for batch in tqdm(test_iter, total=int(len(test)/b_size + 1)):
        batch = conv_f(batch, device)
        y = model.predict(*batch[:-1])
        y.to_cpu()
        pred.append(y.data.ravel())
        
    pred = np.hstack(pred)
    df=test.df.copy()
    df['pred'] = pred
    return df

def classification_summary(result_dir):
    df = pd.read_csv(f'{result_dir}/pred_val.csv')
    t = df['label']
    y = df['pred']
    prec, rec, thresholds = precision_recall_curve(t, y, pos_label=1)
    f1 = 2 * (prec*rec)/(prec+rec)
    threshold = thresholds[f1.argmax()]
    
    df = pd.read_csv(f'{result_dir}/pred_test.csv')
    t = df['label']
    y = df['pred'] > threshold
    prec = precision_score(t, y)
    rec = recall_score(t, y)
    f1 = f1_score(t, y)
    
    result = f'Prec.:{prec*100:.2f}, Rec.:{rec*100:.2f}, F1:{f1*100:.2f}'
    print(result)
    
    with open(f'{result_dir}/score.txt', 'w') as f:
        f.write(result)

    
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='training script for a paraphrase classifier')
    parser.add_argument(
        'model_dir', type=str, help='output directory')
    parser.add_argument(
        '--device', '-d', type=int, default=None, help='gpu device id <int>')
    parser.add_argument(
        '--b_size', '-b', type=int, default=300,
        help='minibatch size <int> (default 300)')
    args = parser.parse_args()
    
    for split in ['val', 'test']:
        df = eval(
            args.model_dir,
            split,
            b_size=args.b_size,
            device=args.device,
            )

        df.to_csv(f'{args.model_dir}/pred_{split}.csv')
    
    classification_summary(args.model_dir)

if __name__ == '__main__':
    main()
