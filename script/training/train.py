import chainer
from chainer import training
from chainer.training import extensions
from chainer import function
import numpy as np
import os
from chainer.iterators import SerialIterator, MultiprocessIterator
import json
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from datetime import datetime as dt
import sys
sys.path.append('./')
from func.datasets.datasets import GTJitterDataset, PLCLCDataset, DDPNDataset
from func.datasets.converters import cvrt_pre_comp_feat
from func.nets.gate_net import Switching_iParaphraseNet, ImageOnlyNet, PhraseOnlyNet, NaiveFuse_iParaphraseNet

chainer.config.multiproc = True  # single proc is faster

def get_model(model_type, gate_mode=None):
    if gate_mode is None:
        if model_type == 'vis':
            model = ImageOnlyNet()
        elif model_type == 'lng':
            model = PhraseOnlyNet()
        return model
            
    if gate_mode == 'off':
        model = NaiveFuse_iParaphraseNet()
    else:
        model = Switching_iParaphraseNet(gate_mode)
        
    return model

def get_data(pl_type, splits, san_check):
    data = []
    for s in splits:
        if pl_type == 'plclc':
            d = PLCLCDataset(s, san_check=san_check)
        elif pl_type == 'ddpn':
            d = DDPNDataset(s, san_check=san_check)
        elif pl_type is None:
            d = DDPNDataset(s, san_check=san_check)
        
        data.append(d)
    
    return data

def train(san_check=False,
          epoch=5,
          lr=0.001,
          b_size=1000,
          device=0,
          w_decay=None,
          out_pref='./checkpoints/',
          model_type='vis+lng',
          pl_type=None,
          gate_mode=None):
    args = locals()

    time_stamp = dt.now().strftime("%Y%m%d-%H%M%S")
    saveto = out_pref + 'sc_' * san_check + time_stamp + '/'
    os.makedirs(saveto)
    json.dump(args, open(saveto + 'args', 'w'))
    print('output to', saveto)
    print('setup dataset...')
    
    train, val = get_data(pl_type, ['train', 'val'], san_check)

    if chainer.config.multiproc:
        train_iter = MultiprocessIterator(train, b_size, n_processes=2)
        val_iter = MultiprocessIterator(
            val, b_size, shuffle=False, repeat=False, n_processes=2)
    else:
        train_iter = SerialIterator(train, b_size)
        val_iter = SerialIterator(val, b_size, shuffle=False, repeat=False)

    print('setup a model: %s' % model_type)
    
    model = get_model(model_type, gate_mode)

    opt = chainer.optimizers.Adam(lr)
    opt.setup(model)

    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    if w_decay:
        opt.add_hook(chainer.optimizer.WeightDecay(w_decay), 'hook_dec')

    updater = training.StandardUpdater(
        train_iter, opt, converter=cvrt_pre_comp_feat, device=device)
    trainer = training.Trainer(updater, (epoch, 'epoch'), saveto)

    val_interval = (1, 'epoch') if san_check else (1000, 'iteration')
    log_interval = (1, 'iteration') if san_check else (10, 'iteration')
    plot_interval = (1, 'iteration') if san_check else (10, 'iteration')
    prog_interval = 1 if san_check else 10

    trainer.extend(
        extensions.Evaluator(
            val_iter, model, converter=cvrt_pre_comp_feat, device=device),
        trigger=val_interval)
    
    if not san_check:
        trainer.extend(
            extensions.ExponentialShift('alpha', 0.5), trigger=(1, 'epoch'))

    # # Comment out to enable visualization of a computational graph.
    trainer.extend(extensions.dump_graph('main/loss'))
    if not san_check:
        ## Comment out next line to save a checkpoint at each epoch, which enable you to restart training loop from the saved point. Note that saving a checkpoint may cost a few minutes.
        trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

        best_val_trigger = training.triggers.MaxValueTrigger(
            'validation/main/f1', trigger=val_interval)
        trainer.extend(
            extensions.snapshot_object(model, 'model'),
            trigger=best_val_trigger)

    # logging extensions
    trainer.extend(
        extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)

    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'main/f1',
            'main/validation/loss', 'validation/main/f1'
        ]),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=log_interval[0]))

    print('start training')
    trainer.run()

    chainer.serializers.save_npz(saveto + 'final_model', model)

    if not san_check:
        return best_val_trigger._best_value


def get_prediction(model_dir, split, device=None):
    model_dir = model_dir + '/' if model_dir[-1] != '/' else model_dir

    settings = json.load(open(model_dir + 'args'))

    print('setup a model ...')
    model_type = settings['model_type']

    model = get_model(model_type)

    chainer.serializers.load_npz(model_dir + 'model', model)

    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    test = get_data(model_type, [split], san_check=False)
    test = test[0]

    test_iter = SerialIterator(
        test, batch_size=1000, repeat=False, shuffle=False)
    conv_f = cvrt_pre_comp_feat

    s_i = 0
    e_i = 0
    pred = np.zeros((len(test), ), dtype=np.float32)

    with function.no_backprop_mode(), chainer.using_config('train', False):

        for i, batch in enumerate(test_iter):
            inputs = conv_f(batch, device)
            score = model.predict(*inputs)
            score.to_cpu()

            e_i = s_i + len(batch)
            pred[s_i:e_i] = score.data.ravel()

            s_i = e_i

    df = test.pair_data.copy()
    df['score'] = pred
    df['ypred'] = pred > .5
    return df


def evaluate(model_dir, split, device=None):
    chainer.config.train = False
    df = get_prediction(model_dir, 'val', device)

    y_true = df.ytrue
    y_pred = df.score

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    best_ind = np.nanargmax(f1)
    best_threshold = thresholds[best_ind]

    print('validation:')
    print('prec: %.2f, rec: %.2f, f1: %.2f' % (precision[best_ind] * 100,
                                               recall[best_ind] * 100,
                                               f1[best_ind] * 100))

    df = get_prediction(model_dir, 'test', device)

    y_true = df.ytrue
    y_pred = df.score > best_threshold
    df['ypred'] = y_pred

    prec = precision_score(y_true, y_pred) * 100
    rec = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    print('test:')
    print('prec: %.2f, rec: %.2f, f1: %.2f' % (prec, rec, f1))

    with open(model_dir + 'res_%s_scores.txt' % split, 'w') as f:
        f.write('prec: %.2f, rec: %.2f, f1: %.2f' % (prec, rec, f1))

    df.to_csv(model_dir + 'res_%s.csv' % split)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='training script for a paraphrase classifier')
    parser.add_argument(
        '--lr', '-lr', type=float, default=0.01, help='learning rate <float>')
    parser.add_argument(
        '--device', '-d', type=int, default=None, help='gpu device id <int>')
    parser.add_argument(
        '--b_size',
        '-b',
        type=int,
        default=500,
        help='minibatch size <int> (default 500)')
    parser.add_argument(
        '--epoch', '-e', type=int, default=5, help='maximum epoch <int>')
    parser.add_argument(
        '--san_check', '-sc', action='store_true', help='sanity check mode')
    parser.add_argument(
        '--w_decay',
        '-wd',
        type=float,
        default=None,
        help='weight decay <float>')
    parser.add_argument(
        '--settings', type=str, default=None, help='path to arg file')
    parser.add_argument('--model_type', '-mt', default='vis+lng')
    parser.add_argument('--pl_type', '-pt', default='ddpn')
    parser.add_argument('--gate_mode', '-gm', default='off')
    parser.add_argument(
        '--eval',
        type=str,
        default=None,
        help='path to an output directory <str>. the model will be evaluated.')
    parser.add_argument('--out_pref', type=str, default='./checkpoint/')
    args = parser.parse_args()

    if args.eval is not None:
        evaluate(args.eval, split='test', device=args.device)
    else:
        args_dic = vars(args)

        if args.settings:
            settings = args.settings
            epoch = args.epoch
            device = args.device
            out_pref = args.out_pref

            prev_args = json.load(
                open(os.path.join(os.path.dirname(args.settings), 'args')))
            args_dic.update(prev_args)
            args_dic['settings'] = settings
            args_dic['epoch'] = epoch
            args_dic['device'] = device
            args_dic['out_pref'] = out_pref

        train(
            san_check=args_dic['san_check'],
            epoch=args_dic['epoch'],
            lr=args_dic['lr'],
            b_size=args_dic['b_size'],
            device=args_dic['device'],
            w_decay=args_dic['w_decay'],
            model_type=args_dic['model_type'],
            pl_type=args_dic['pl_type'],
            gate_mode=args_dic['gate_mode'],
            out_pref=args_dic['out_pref'])


if __name__ == '__main__':
    main()
