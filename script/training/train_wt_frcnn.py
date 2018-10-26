import matplotlib
matplotlib.use('Agg')
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
from chainer.dataset.convert import concat_examples
import pandas as pd
from collections import defaultdict
from chainer import dataset
from imageio import imread
from chainer import initializers
from chainer.dataset import iterator
from chainer.dataset.convert import to_device
import _pickle as pickle
import json
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from datetime import datetime as dt
import tables
from chainercv.utils import bbox_iou
import random
from functools import reduce
from chainer import reporter
import cupy
import shutil
import sys
sys.path.append('./')
from func.nets.faster_rcnn import FasterRCNNExtractor
from func.datasets.datasets import PreCompFeatDataset

chainer.config.multiproc = True  # single proc is faster

import chainer
import chainer.functions as F
from chainer.iterators import SerialIterator
import pandas as pd
import tables
import numpy as np
import imageio
import os

# class Dataset(chainer.dataset.DatasetMixin):
#     def __init__(self, split, san_check=False):
#         pair_data = pd.read_csv('data/phrase_pair_%s.csv' % split)

#         if san_check:
#             skip = 2000 if split == 'train' else 1000
#             pair_data = pair_data.iloc[::skip]

#         self.pair_data = pair_data

#         self.gtroi_data = pd.read_csv('data/gt_roi_cord_%s.csv' % split)
#         self.img_root = 'data/flickr30k-images/'

#         # get phrse indices
#         p2i_dict = defaultdict(lambda: -1)
#         with open('data/phrase_misc/%s_uniquePhrases' % split) as f:
#             for i, line in enumerate(f):
#                 p2i_dict[line.rstrip()] = i

#         self._p2i_dict = p2i_dict
#         self._feat = np.load('data/phrase_feat/wea/%s.npy'%split)

#         print('%s data: %i pairs' % (split, len(self.pair_data)))

#     def __len__(self):
#         return len(self.pair_data)

#     def get_phrases(self, i):
#         return self.pair_data.iloc[i][['phrase1', 'phrase2']]

#     def get_phrase_feat(self, i):
#         phr1, phr2 = self.get_phrases(i)
#         x1 = self._feat[self._p2i_dict[phr1]]
#         x2 = self._feat[self._p2i_dict[phr2]]
#         return x1, x2

#     def get_gt_roi(self, i):
#         img_id, phr_1, phr_2 = self.pair_data.iloc[i][['image', 'phrase1', 'phrase2']]
#         rois = self.gtroi_data[self.gtroi_data.image == img_id]

#         gt_rois = []
#         for phr in [phr_1, phr_2]:
#             roi = rois[rois.phrase == phr][['ymin', 'xmin', 'ymax', 'xmax']]
#             gt_roi_min = roi.min(axis=0)
#             gt_roi_max = roi.max(axis=0)
#             roi = np.hstack((gt_roi_min[:2], gt_roi_max[-2:]))
#             gt_rois.append(roi)

#         return gt_rois[0], gt_rois[1]

#     def read_image(self, i):
#         img_id = self.pair_data.iloc[i][['image']].values[0]
#         img = imageio.imread(os.path.join(self.img_root, str(img_id))+'.jpg')
#         return img

#     def get_example(self, i):
#         img = self.read_image(i)
#         phr_1, phr_2 = self.get_phrase_feat(i)
#         gt_roi_1, gt_roi_2 = self.get_gt_roi(i)
#         l = self.pair_data.iloc[i][['ytrue']].values[0]

#         return img, phr_1, phr_2, gt_roi_1, gt_roi_2, l

# def my_converter(batch, device=None):
#     img = [b[0].transpose(2, 0, 1) for b in batch]
#     phr_1 = np.vstack([b[1] for b in batch]).astype('f')
#     phr_2 = np.vstack([b[2] for b in batch]).astype('f')
#     gtroi_1 = np.vstack([b[3] for b in batch]).astype('f')
#     gtroi_2 = np.vstack([b[4] for b in batch]).astype('f')
#     l = np.asarray([b[5] for b in batch]).astype('i')

#     if chainer.config.train:
#         img_size = np.asarray([x.shape for x in img])
#         gtroi_1 = jitter_bbox(gtroi_1, img_size)
#         gtroi_2 = jitter_bbox(gtroi_2, img_size)

#     if device is not None:
#         # img = [to_device(device, x) for x in img]
#         phr_1 = to_device(device, phr_1)
#         phr_2 = to_device(device, phr_2)
#         gtroi_1 = to_device(device, gtroi_1)
#         gtroi_2 = to_device(device, gtroi_2)
#         l = to_device(device, l)

#     return img, phr_1, phr_2, gtroi_1, gtroi_2, l


def my_converter(batch, device=None):
    phr_1 = np.vstack([b[0] for b in batch]).astype('f')
    phr_2 = np.vstack([b[1] for b in batch]).astype('f')
    xvis_1 = np.vstack([b[2] for b in batch]).astype('f')
    xvis_2 = np.vstack([b[3] for b in batch]).astype('f')
    l = np.asarray([b[4] for b in batch]).astype('i')

    if device is not None:
        phr_1 = to_device(device, phr_1)
        phr_2 = to_device(device, phr_2)
        gtroi_1 = to_device(device, xvis_1)
        gtroi_2 = to_device(device, xvis_2)
        l = to_device(device, l)

    return phr_1, phr_2, xvis_1, xvis_2, l


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
        h = F.squeeze(h)
        loss = F.sigmoid_cross_entropy(h, L)

        if chainer.config.train == False:
            self.y = F.sigmoid(h)
            self.t = L

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
            self.base_net = FasterRCNNExtractor(
                n_fg_class=20, pretrained_model='voc07')
            self.fusion_net = FusionNet()
            self.classifier = ClassifierNet()

    def predict(self, ):
        pass

    def extract_visfeat(self, img, gtroi_1, gtroi_2):
        img_ = [self.base_net.prepare(x) for x in img]
        scale = [
            max(im_.shape[1:]) / max(im.shape[1:])
            for im, im_ in zip(img, img_)
        ]
        img_ = [to_device(0, x) for x in img_]

        y = []

        roi_indices = self.xp.zeros((2, )).astype('i')
        for im, roi1, roi2, s in zip(img_, gtroi_1, gtroi_2, scale):
            roi = self.xp.vstack((roi1, roi2)) * s
            y.append(self.base_net.extract(im[None, :], roi, roi_indices))

        y = F.vstack(y)
        y = y.reshape(-1, 4096 * 2)

        return y

    def __call__(self, img, phr_1, phr_2, gtroi_1, gtroi_2, l):
        with chainer.no_backprop_mode():
            y = self.extract_visfeat(img, gtroi_1, gtroi_2)
            y1, y2 = F.split_axis(y, 2, axis=-1)

        h1 = self.fusion_net(y1, phr_1)
        h2 = self.fusion_net(y2, phr_2)

        loss = self.classifier(h1, h2, l)
        return loss


class iOnlyNet(chainer.Chain):
    def __init__(self,
                 projection_net,
                 classifier_net,
                 attention_net=None,
                 kl_on=False,
                 alpha=None):
        super(iOnlyNet, self).__init__()
        self.use_gt_attention = (attention_net is None)

        with self.init_scope():
            self.projection_net = projection_net
            self.classifier_net = classifier_net
            if attention_net is not None:
                self.attention_net = attention_net

        self._kl_on = kl_on
        self.alpha = alpha

    def select_gt_region(self, region_feats, region_label):
        x = [
            F.embed_id(i, W)
            for i, W in zip(region_label[:, None], region_feats)
        ]
        return F.vstack(x)

    def compute_weigted_feat(self, region_feats, att):
        att = F.expand_dims(att, axis=-1)
        region_feats = region_feats * F.broadcast_to(att, region_feats.shape)
        return F.sum(region_feats, axis=1)

    def predict(self, Xr, Xp1, Xp2, Lr1, Lr2, L):
        if self.use_gt_attention:
            hr1 = self.select_gt_region(Xr, Lr1)
            hr2 = self.select_gt_region(Xr, Lr2)
            l_att = 0
        else:
            att1, l_att1 = self.attention_net(Xp1, Xr, Lr1)
            att2, l_att2 = self.attention_net(Xp2, Xr, Lr2)
            hr1 = self.compute_weigted_feat(Xr, att1)
            hr2 = self.compute_weigted_feat(Xr, att2)
            l_att = (l_att1 + l_att2) * .5

        h1 = self.projection_net(hr1)
        h2 = self.projection_net(hr2)

        _ = self.classifier_net(h1, h2, L)
        return self.classifier_net.y, self.classifier_net.t

    def __call__(self, Xr, Xp1, Xp2, Lr1, Lr2, L):
        if self.use_gt_attention:
            hr1 = self.select_gt_region(Xr, Lr1)
            hr2 = self.select_gt_region(Xr, Lr2)
            l_att = 0
        else:
            att1, l_att1 = self.attention_net(Xp1, Xr, Lr1)
            att2, l_att2 = self.attention_net(Xp2, Xr, Lr2)
            hr1 = self.compute_weigted_feat(Xr, att1)
            hr2 = self.compute_weigted_feat(Xr, att2)
            l_att = (l_att1 + l_att2) * .5

        h1 = self.projection_net(hr1)
        h2 = self.projection_net(hr2)

        l_cls = self.classifier_net(h1, h2, L)

        if self._kl_on:
            l_kl = kl_loss(att1, att2, L, alpha=self.alpha)
            loss = l_cls + l_att + l_kl
            reporter.report({'loss': loss, 'kl_loss': l_kl}, self)
        else:
            loss = l_cls + l_att
            reporter.report({'loss': loss}, self)

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


def train(
        san_check=False,
        epoch=5,
        lr=0.001,
        dr_ratio=.4,
        b_size=500,
        device=0,
        w_decay=None,
        out_pref='./checkpoints/',
        # resume='',
        alpha=None):
    args = locals()

    out_base = out_pref + '%s_' % ('gtroi_jittering')
    time_stamp = dt.now().strftime("%Y%m%d-%H%M%S")
    saveto = out_base + 'sc_' * san_check + time_stamp + '/'
    os.makedirs(saveto)
    json.dump(args, open(saveto + 'args', 'w'))
    print('output to', saveto)
    print('setup dataset...')

    train = Dataset('train', san_check=san_check)
    val = Dataset('val', san_check=san_check)

    if chainer.config.multiproc:
        train_iter = MultiprocessIterator(train, b_size, n_processes=2)
        val_iter = MultiprocessIterator(
            val, b_size, shuffle=False, repeat=False, n_processes=2)
    else:
        train_iter = SerialIterator(train, b_size)
        val_iter = SerialIterator(val, b_size, shuffle=False, repeat=False)

    print('setup a model ...')
    model = iParaphraseNet()
    opt = chainer.optimizers.Adam(lr)
    opt.setup(model)

    # # set updaterules for attention net
    # for path, param in model.namedparams():
    #     if path.split('/')[1] == 'attention_net':
    #         param.update_rule.hyperparam.alpha = lr_att
    #         # param.update_rule = chainer.update_rules.MomentumSGD(lr_att).create_update_rule()

    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    if w_decay:
        opt.add_hook(chainer.optimizer.WeightDecay(w_decay), 'hook_dec')

    updater = training.StandardUpdater(
        train_iter, opt, converter=my_converter, device=device)
    trainer = training.Trainer(updater, (epoch, 'epoch'), saveto)

    val_interval = (1, 'epoch') if san_check else (500, 'iteration')
    log_interval = (1, 'iteration') if san_check else (10, 'iteration')
    plot_interval = (1, 'iteration') if san_check else (10, 'iteration')
    prog_interval = 1 if san_check else 10

    trainer.extend(
        extensions.Evaluator(
            val_iter, model, converter=my_converter, device=device),
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
    # logged_links = [
    #     model.fusion_net.fuse_p,
    #     model.fusion_net.fuse_r2
    # ]
    # statistics = {
    #     'min': cupy.min,
    #     'max': cupy.max,
    # }
    # trainer.extend(extensions.ParameterStatistics(logged_links, statistics, trigger=log_interval))

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

    mode = settings['mode']
    v_feat_type, p_feat_type = settings['intype'].split('+')

    print('setup a model ...')
    model = setup_model(mode, dr_ratio=0.0)
    chainer.serializers.load_npz(model_dir + 'model', model)

    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    test, _ = get_dataset(
        split, mode=mode, v_feat=v_feat_type, p_feat=p_feat_type)
    test_iter = SerialIterator(
        test, batch_size=300, repeat=False, shuffle=False)
    conv_f = concat_examples

    s_i = 0
    e_i = 0
    pred = np.zeros((len(test), ), dtype=np.float32)

    with function.no_backprop_mode(), chainer.using_config('train', False):

        for i, batch in enumerate(test_iter):
            inputs = conv_f(batch, device)
            score, _ = model.predict(*inputs)
            score.to_cpu()

            e_i = s_i + len(batch)
            pred[s_i:e_i] = score.data.ravel()

            s_i = e_i

    df = pd.DataFrame({
        'image': test._image_id,
        'phrase1': test._phrase1,
        'phrase2': test._phrase2,
        'ytrue': test._label,
        'score': pred,
        'ypred': pred > .5,
    })

    return df


def evaluate(model_dir, split, device=None):
    chainer.config.train = False
    df = get_prediction(model_dir, 'val', device)

    y_true = df.ytrue
    y_pred = df.score

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[f1.argmax()]

    df = get_prediction(model_dir, 'test', device)

    y_true = df.ytrue
    y_pred = df.score > best_threshold

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
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
        default=20,
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
    parser.add_argument(
        '--eval',
        type=str,
        default=None,
        help='path to an output directory <str>. the model will be evaluated.')
    parser.add_argument('--out_pref', type=str, default='./')
    args = parser.parse_args()

    if args.eval is not None:
        # evaluate(args.eval, split='val', device=args.device)
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
            out_pref=args_dic['out_pref'])


if __name__ == '__main__':
    main()
