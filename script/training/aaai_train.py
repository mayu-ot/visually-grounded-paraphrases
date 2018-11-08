import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
import numpy as np
import os
from chainer.iterators import SerialIterator, MultiprocessIterator
from chainer import function
from chainer import cuda
import pandas as pd
from chainer import initializers
import json
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from datetime import datetime as dt
from chainer import reporter

import sys
sys.path.append('./')
from func.datasets.datasets import GTJitterDataset, PLCLCDataset, DDPNDataset
from func.datasets.converters import cvrt_pre_comp_feat

chainer.config.multiproc = True  # single proc is faster

def binary_classification_summary(y, t):
    xp = cuda.get_array_module(y)
    y = y.data

    y = y.ravel()
    true = t.ravel()
    pred = (y > 0)
    support = xp.sum(true)

    gtp_mask = xp.where(true)
    relevant = xp.sum(pred)
    tp = pred[gtp_mask].sum()

    if (support == 0) or (relevant == 0) or (tp == 0):
        return xp.array(0.), xp.array(0.), xp.array(0.)

    prec = tp * 1. / relevant
    recall = tp * 1. / support
    f1 = 2. * (prec * recall) / (prec + recall)

    return prec, recall, f1


class PhraseProjectionNet(chainer.Chain):
    def __init__(self):
        super(PhraseProjectionNet, self).__init__()
        with self.init_scope():
            self.setup_layers()

    def setup_layers(self):
        h_size = 1000
        # fusenet for phrase and region features
        self.fuse_p = L.Linear(
            None, h_size, initialW=initializers.HeNormal(), nobias=True)
        self.bn_2 = L.BatchNormalization(h_size)
        self.fuse_2 = L.Linear(
            None, 300, initialW=initializers.HeNormal(), nobias=True)
        self.bn_3 = L.BatchNormalization(300)

    def __call__(self, Xp):
        h = F.relu(self.bn_2(self.fuse_p(Xp)))
        h = F.relu(self.bn_3(self.fuse_2(h)))
        return h


class ImageProjectionNet(chainer.Chain):
    def __init__(self):
        super(ImageProjectionNet, self).__init__()
        with self.init_scope():
            self.setup_layers()

    def setup_layers(self):
        h_size = 1000
        # fusenet for phrase and region features
        self.fuse_r1 = L.Linear(
            None, h_size, initialW=initializers.HeNormal(), nobias=True)
        self.bn_1 = L.BatchNormalization(h_size)
        self.fuse_r2 = L.Linear(
            None, h_size, initialW=initializers.HeNormal(), nobias=True)
        self.bn_2 = L.BatchNormalization(h_size)
        self.fuse_2 = L.Linear(
            None, 300, initialW=initializers.HeNormal(), nobias=True)
        self.bn_3 = L.BatchNormalization(300)

    def __call__(self, Xvis):
        Xvis = F.relu(self.bn_1(self.fuse_r1(Xvis)))
        h = F.relu(self.bn_2(self.fuse_r2(Xvis)))
        h = F.relu(self.bn_3(self.fuse_2(h)))
        return h


class FusionNet(chainer.Chain):
    def __init__(self):
        super(FusionNet, self).__init__()
        with self.init_scope():
            self.setup_layers()

    def setup_layers(self):
        h_size = 1000
        # fusenet for phrase and region features
        self.fuse_r1 = L.Linear(
            None, h_size, initialW=initializers.HeNormal(), nobias=True)
        self.bn_1 = L.BatchNormalization(h_size)
        self.fuse_r2 = L.Linear(
            None, h_size, initialW=initializers.HeNormal(), nobias=True)
        self.fuse_p = L.Linear(
            None, h_size, initialW=initializers.HeNormal(), nobias=True)
        self.bn_2 = L.BatchNormalization(h_size)
        self.fuse_2 = L.Linear(
            None, 300, initialW=initializers.HeNormal(), nobias=True)
        self.bn_3 = L.BatchNormalization(300)

    def __call__(self, Xvis, Xp):
        Xvis = F.relu(self.bn_1(self.fuse_r1(Xvis)))
        h = F.relu(self.bn_2(self.fuse_r2(Xvis) + self.fuse_p(Xp)))
        h = F.relu(self.bn_3(self.fuse_2(h)))
        return h


class ClassifierNet(chainer.Chain):
    def __init__(self, dr_ratio=.4):
        self._dr_ratio = dr_ratio
        super(ClassifierNet, self).__init__()
        with self.init_scope():
            self.mlp_1 = L.Linear(
                None, 128, initialW=initializers.HeNormal(), nobias=True)
            self.mlp_2 = L.Linear(
                None, 128, initialW=initializers.HeNormal(), nobias=True)
            self.bn_4 = L.BatchNormalization(128)
            self.cls = L.Linear(None, 1, initialW=initializers.LeCunNormal())

    def __call__(self, X1, X2, L):
        # paraphrase classification

        h = F.relu(self.bn_4(self.mlp_1(X1) + self.mlp_2(X2)))
        h = self.cls(F.dropout(h, self._dr_ratio))
        h = F.flatten(h)

        if chainer.config.train == False:
            self.y = F.sigmoid(h)
            self.t = L

        loss = F.sigmoid_cross_entropy(h, L)

        precision, recall, fbeta = binary_classification_summary(h, L)
        reporter.report({
            'loss': loss,
            'precision': precision,
            'recall': recall,
            'f1': fbeta
        }, self)
        return loss


class iParaphraseNet(chainer.Chain):
    def __init__(self):
        super(iParaphraseNet, self).__init__()
        with self.init_scope():
            self.fusion_net = FusionNet()
            self.classifier = ClassifierNet()

    def predict(self, phr_1, phr_2, xvis_1, xvis_2, l):
        _ = self(phr_1, phr_2, xvis_1, xvis_2, l)
        y = self.classifier.y
        return y

    def __call__(self, phr_1, phr_2, xvis_1, xvis_2, l):

        h1 = self.fusion_net(xvis_1, phr_1)
        h2 = self.fusion_net(xvis_2, phr_2)

        loss = self.classifier(h1, h2, l)
        return loss


class iOnlyNet(chainer.Chain):
    def __init__(self):
        super(iOnlyNet, self).__init__()
        with self.init_scope():
            self.proj_net = ImageProjectionNet()
            self.classifier = ClassifierNet()

    def predict(self, phr_1, phr_2, xvis_1, xvis_2, l):
        _ = self(phr_1, phr_2, xvis_1, xvis_2, l)
        y = self.classifier.y
        return y

    def __call__(self, phr_1, phr_2, xvis_1, xvis_2, l):
        h1 = self.proj_net(xvis_1)
        h2 = self.proj_net(xvis_2)
        loss = self.classifier(h1, h2, l)

        return loss


class pOnlyNet(chainer.Chain):
    def __init__(self):
        super(pOnlyNet, self).__init__()
        with self.init_scope():
            self.proj_net = PhraseProjectionNet()
            self.classifier = ClassifierNet()

    def predict(self, phr_1, phr_2, xvis_1, xvis_2, l):
        _ = self(phr_1, phr_2, xvis_1, xvis_2, l)
        y = self.classifier.y
        return y

    def __call__(self, phr_1, phr_2, xvis_1, xvis_2, l):

        h1 = self.proj_net(phr_1)
        h2 = self.proj_net(phr_2)
        loss = self.classifier(h1, h2, l)

        return loss


def postprocess(res):
    keys = list(res.keys())

    for k in keys:
        if k == 'main/classifier_net/loss':
            res['main/cls_loss'] = res[k]
        elif k == 'main/classifier_net/f1':
            res['main/f1'] = res[k]
        elif k == 'validation/main/classifier_net/f1':
            res['validation/main/f1'] = res[k]
        elif k == 'validation/main/classifier_net/loss':
            res['validation/main/cls_loss'] = res[k]
        else:
            pass


def train(san_check=False,
          epoch=5,
          lr=0.001,
          dr_ratio=.4,
          b_size=1000,
          device=0,
          w_decay=None,
          out_pref='./checkpoints/',
          model_type='vis+lng',
          alpha=None):
    args = locals()

    time_stamp = dt.now().strftime("%Y%m%d-%H%M%S")
    saveto = out_pref + 'sc_' * san_check + time_stamp + '/'
    os.makedirs(saveto)
    json.dump(args, open(saveto + 'args', 'w'))
    print('output to', saveto)
    print('setup dataset...')

    if model_type in ['vis+lng+gtroi', 'vis+gtroi']:
        train = GTJitterDataset('train', san_check=san_check)
        val = GTJitterDataset('val', san_check=san_check)
    elif model_type in ['vis+lng+plclcroi', 'vis+plclcroi']:
        train = PLCLCDataset('train', san_check=san_check)
        val = PLCLCDataset('val', san_check=san_check)
    elif model_type in ['vis+lng+ddpnroi', 'vis+ddpnroi']:
        train = DDPNDataset('train', san_check=san_check) 
        val = DDPNDataset('val', san_check=san_check)
    else:
        train = GTJitterDataset('train', san_check=san_check)
        val = PLCLCDataset(
            'val', san_check=san_check)  # validate on plclc bbox

    if chainer.config.multiproc:
        train_iter = MultiprocessIterator(train, b_size, n_processes=2)
        val_iter = MultiprocessIterator(
            val, b_size, shuffle=False, repeat=False, n_processes=2)
    else:
        train_iter = SerialIterator(train, b_size)
        val_iter = SerialIterator(val, b_size, shuffle=False, repeat=False)

    print('setup a model: %s' % model_type)

    if model_type in [
            'vis+lng', 'vis+lng+gtroi', 'vis+lng+plclcroi', 'vis+lng+ddpnroi'
    ]:
        model = iParaphraseNet()
    elif model_type in ['vis', 'vis+gtroi', 'vis+plclcroi', 'vis+ddpnroi']:
        model = iOnlyNet()
    elif model_type == 'lng':
        model = pOnlyNet()
    else:
        raise RuntimeError('invalid model_type: %s' % model_type)

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
    # trainer.extend(extensions.dump_graph('main/loss'))
    if not san_check:
        ## Comment out next line to save a checkpoint at each epoch, which enable you to restart training loop from the saved point. Note that saving a checkpoint may cost a few minutes.
        trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

        best_val_trigger = training.triggers.MaxValueTrigger(
            'validation/main/classifier/f1', trigger=val_interval)
        trainer.extend(
            extensions.snapshot_object(model, 'model'),
            trigger=best_val_trigger)

    # logging extensions
    trainer.extend(
        extensions.LogReport(trigger=log_interval, postprocess=postprocess))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)

    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'main/classifier/loss',
            'main/validation/classifier/loss'
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

    if model_type in [
            'vis+lng', 'vis+lng+gtroi', 'vis+lng+plclcroi', 'vis+lng+ddpnroi'
    ]:
        model = iParaphraseNet()
    elif model_type in ['vis', 'vis+gtroi', 'vis+plclcroi', 'vis+ddpnroi']:
        model = iOnlyNet()
    elif model_type == 'lng':
        model = pOnlyNet()
    else:
        raise RuntimeError('invalid model_type: %s' % model_type)

    chainer.serializers.load_npz(model_dir + 'model', model)

    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    if model_type in ['vis+lng+gtroi', 'vis+gtroi']:
        test = GTJitterDataset('test', san_check=False)
    elif model_type in ['vis+lng+ddpnroi', 'vis+ddpnroi']:
        test = DDPNDataset('test', san_check=False)
    else:
        test = PLCLCDataset(split, san_check=False)

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
    print('prec: %.4f, rec: %.4f, f1: %.4f' % (precision[best_ind],
                                               recall[best_ind], f1[best_ind]))

    df = get_prediction(model_dir, 'test', device)

    y_true = df.ytrue
    y_pred = df.score > best_threshold
    df['ypred'] = y_pred

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print('test:')
    print('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))

    with open(model_dir + 'res_%s_scores.txt' % split, 'w') as f:
        f.write('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))

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
        '--dr_ratio',
        '-dr',
        type=float,
        default=0.0,
        help='dropout ratio <float>')
    parser.add_argument(
        '--settings', type=str, default=None, help='path to arg file')
    parser.add_argument('--model_type', '-mt', default='vis+lng')
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
            dr_ratio=args_dic['dr_ratio'],
            b_size=args_dic['b_size'],
            device=args_dic['device'],
            w_decay=args_dic['w_decay'],
            model_type=args_dic['model_type'],
            out_pref=args_dic['out_pref'])


if __name__ == '__main__':
    main()
