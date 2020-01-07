import chainer
from chainer.dataset import DatasetMixin
from chainer import training
from chainer.training import extensions
import os
from chainer.iterators import SerialIterator
import json
from datetime import datetime as dt
import sys
sys.path.append('./')

import numpy as np
import pandas as pd

from dataset import Dataset, conv_f
from snn import VGPNet

# chainer.config.multiproc = True  # single proc is faster

def resample_data(trainer):
    trainer.updater._iterators['main'].dataset.downsample()


def train(
          method,
          san_check=False,
          epoch=5,
          lr=0.001,
          b_size=500,
          device=0,
          w_decay=0.0001,
          out_pref='./checkpoints/'
         ):
    args = locals()

    time_stamp = dt.now().strftime("%Y%m%d-%H%M%S")
    saveto = out_pref + 'sc_' * san_check + time_stamp + '/'
    os.makedirs(saveto)
    json.dump(args, open(saveto + 'args', 'w'))
    print('output to', saveto)
    print('setup dataset...')

    train = Dataset(method, 'train', san_check=san_check)
    val = Dataset(method, 'val', san_check=san_check)
    

    train_iter = SerialIterator(train, b_size)
    val_iter = SerialIterator(val, b_size*2, shuffle=False, repeat=False)
    
    emb_W = np.load('data/processed/word2vec.npy').astype('f')
    model = VGPNet(128, emb_W)
    
    opt = chainer.optimizers.Adam(lr)
    opt.setup(model)

    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    if w_decay:
        opt.add_hook(chainer.optimizer.WeightDecay(w_decay), 'hook_dec')

    updater = training.StandardUpdater(
        train_iter, opt, converter=conv_f, device=device)
    trainer = training.Trainer(updater, (epoch, 'epoch'), saveto)

    val_interval = (1, 'epoch')
    log_interval = (1, 'iteration') if san_check else (10, 'iteration')
    plot_interval = (1, 'iteration') if san_check else (10, 'iteration')
    prog_interval = 1 if san_check else 10

    trainer.extend(
            extensions.ExponentialShift('alpha', 0.5), trigger=(1, 'epoch'))

    # logging extensions
    trainer.extend(
        extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)

    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'main/f1',
            'validation/main/loss', 'validation/main/f1'
        ]),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=log_interval[0]))
    
    trainer.extend(
        extensions.Evaluator(
            val_iter, model, converter=conv_f, device=device),
        trigger=val_interval)

    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    best_val_trigger = training.triggers.MaxValueTrigger(
            'validation/main/f1', trigger=val_interval)
    trainer.extend(
            extensions.snapshot_object(model, 'model'),
            trigger=best_val_trigger)
    trainer.extend(resample_data, trigger=(1, 'epoch'))

    print('start training')
    trainer.run()

    chainer.serializers.save_npz(saveto + 'final_model', model)

    if not san_check:
        return best_val_trigger._best_value


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='training script for a paraphrase classifier')
    parser.add_argument('localization_method', type=str,
                       help='ddpn or plclc')
    parser.add_argument(
        '--lr', '-lr', type=float, default=0.01, help='learning rate <float>')
    parser.add_argument(
        '--device', '-d', type=int, default=3, help='gpu device id <int>')
    parser.add_argument(
        '--b_size', '-b', type=int, default=500,
        help='minibatch size <int> (default 500)')
    parser.add_argument(
        '--epoch', '-e', type=int, default=5, help='maximum epoch <int>')
    parser.add_argument(
        '--san_check', '-sc', action='store_true', help='sanity check mode')
    parser.add_argument(
        '--w_decay',
        '-wd',
        type=float,
        default=0.0001,
        help='weight decay <float>')
    parser.add_argument('--out_pref', type=str, default='./models/tpami/SNN+DDPN/')
    args = parser.parse_args()

    args_dic = vars(args)

    train(
        method=args_dic['localization_method'],
        san_check=args_dic['san_check'],
        epoch=args_dic['epoch'],
        lr=args_dic['lr'],
        b_size=args_dic['b_size'],
        device=args_dic['device'],
        w_decay=args_dic['w_decay'],
        out_pref=args_dic['out_pref'])


if __name__ == '__main__':
    main()
