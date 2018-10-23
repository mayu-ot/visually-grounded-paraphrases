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
from imageio import  imread
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

chainer.config.multiproc = False # single proc is faster

class EntityDatasetBase(dataset.DatasetMixin):
    def __init__(self, data_file, san_check=False, skip=None):
        print(data_file)
        df = pd.read_csv(data_file)

        self._image_id = df.image.values
        self._phrase1 = df.phrase1.values
        self._phrase2 = df.phrase2.values
        self._label = df.ytrue.values
        self._roi_label = df[['roi1', 'roi2']].values
        
        if san_check:
            print('sanity chack mode')
            self._image_id = self._image_id[::2000]
            self._phrase1 = self._phrase1[::2000]
            self._phrase2 = self._phrase2[::2000]
            self._label = self._label[::2000]
            self._roi_label = self._roi_label[::2000]

        if skip is not None:
            print('sample data with skip %i'%skip)
            self._image_id = self._image_id[::skip]
            self._phrase1 = self._phrase1[::skip]
            self._phrase2 = self._phrase2[::skip]
            self._label = self._label[::skip]

    def __len__(self):
        return len(self._label)
    
    def _get_entity(self, i):
        raise NotImplementedError
    
    def _get_label(self, i):
        return self._label[i]

    def _get_example(self, i):
        
        # get phrase feature
        x1, x2 = self._get_entity(i)
        
        # get label
        y = self._get_label(i)
        y = np.asarray(y, dtype=np.int32)
        
        return x1, x2, y
    
    def get_example(self, i):
        raise NotImplementedError

def save_wea():
    word_dict = pickle.load(open('data/phrase_misc/word_dict', 'rb'), encoding='latin1')
    word_vec = np.load('data/phrase_misc/word_vec.npy')

    for split in ['train', 'val', 'test']:    
        with open('data/phrase_misc/%s_uniquePhrases' % split) as f:
            for i, _ in enumerate(f):
                pass
        print('%i phrases' % (i + 1))
        X = np.zeros((i + 1, 300), dtype=np.float32)

        with open('data/phrase_misc/%s_uniquePhrases' % split) as f:
            for i, line in enumerate(f):
                words = line.rstrip().split('+')
                word_ids = [word_dict[w] for w in words]
                wea_feat = np.mean(word_vec[word_ids], axis=0)
                X[i] = wea_feat
        
        np.save('data/phrase_feat/wea/%s' % split, X)

class RegionEntityDatasetBase(EntityDatasetBase):
    def __init__(self, data_file, img_data_file, san_check=False, preload=False, skip=None):
        super(RegionEntityDatasetBase, self).__init__(data_file, san_check=san_check, skip=skip)
        
        self._h5file = img_data_file
        if not chainer.config.multiproc:
            print('use single process data provider')
            self._f = tables.open_file(self._h5file)
    
    def __del__(self):
        if not chainer.config.multiproc:
            self._f.close()

    def _get_image(self, i):
        if chainer.config.multiproc:
            with tables.open_file(self._h5file) as f:
                feat = f.get_node('/', str(self._image_id[i])).read()
        else:
            feat = self._f.get_node('/', str(self._image_id[i])).read()
        return feat
    
    def get_example(self, i):
        # get image
        image = self._get_image(i)
        
        # get phrase feature
        x1, x2 = self._get_entity(i)
        
        # get label
        y = self._get_label(i)
        y = np.asarray(y, np.int32)
        
        return image, x1, x2, y

class EntityDatasetPhraseFeat(EntityDatasetBase):
    def __init__(self, data_file, phrase_feature_file, unique_phrase_file, san_check=False, preload=False):
        super(EntityDatasetPhraseFeat, self).__init__(data_file, san_check=san_check)
    
        phrase2id_dict = defaultdict(lambda: -1)
        with open(unique_phrase_file) as f:
            for i, line in enumerate(f):
                phrase2id_dict[line.rstrip()] = i
        
        self._p2i_dict = phrase2id_dict
        self._feat = np.load(phrase_feature_file).astype(np.float32)

    def _get_entity(self, i):
        x1 = self._feat[self._p2i_dict[self._phrase1[i]]]
        x2 = self._feat[self._p2i_dict[self._phrase2[i]]]
        return x1, x2

    def get_example(self, i):
        # get phrase feature
        x1, x2 = self._get_entity(i)
        
        # get label
        y = self._get_label(i)
        y = np.asarray(y, np.int32)
        
        return x1, x2, y

class RegionEntityDatasetPhraseFeat(RegionEntityDatasetBase):
    def __init__(self, data_file, phrase_feature_file, unique_phrase_file, img_data_file, san_check=False, preload=False, skip=None):
        super(RegionEntityDatasetPhraseFeat, self).__init__(data_file, img_data_file, san_check=san_check, preload=preload, skip=skip)
        
        phrase2id_dict = defaultdict(lambda: -1)
        with open(unique_phrase_file) as f:
            for i, line in enumerate(f):
                phrase2id_dict[line.rstrip()] = i
        
        self._p2i_dict = phrase2id_dict
        self._feat = np.load(phrase_feature_file).astype(np.float32)

    def _get_entity(self, i):
        x1 = self._feat[self._p2i_dict[self._phrase1[i]]]
        x2 = self._feat[self._p2i_dict[self._phrase2[i]]]
        return x1, x2

class RegionEntityDatasetPhraseFeatwtROIWeights(RegionEntityDatasetPhraseFeat):
    
    def __init__(self,
                 data_file,
                 phrase_feature_file,
                 unique_phrase_file,
                 img_data_file,
                 san_check=False):

        super(RegionEntityDatasetPhraseFeatwtROIWeights, self).__init__(
            data_file, phrase_feature_file, unique_phrase_file, img_data_file, san_check=san_check)
    
    def _get_roi_label(self, i):
        l1, l2 = self._roi_label[i]
        return l1, l2

    
    def get_example(self, i):
        # get image
        image = self._get_image(i)
        
        # get phrase feature
        x1, x2 = self._get_entity(i)

        gt_r1, gt_r2 = self._get_roi_label(i)
        
        # get label
        y = self._get_label(i)
        y = np.asarray(y, dtype=np.int32)
        
        return image, x1, x2, gt_r1, gt_r2, y

class RegionEntityDatasetGT(EntityDatasetPhraseFeat):

    def __init__(self, data_file,
                 phrase_feature_file,
                 unique_phrase_file,
                 img_data_file,
                 san_check=False,
                 preload=False,
                 skip=None):
        
        super(RegionEntityDatasetGT, self).__init__(data_file, phrase_feature_file, unique_phrase_file, san_check, preload=preload)
        img_feat_f, img_index_f = img_data_file
        
        self.im_feat = np.load(img_feat_f)
        self.feat_indices = np.load(img_index_f)
        # self.roi_meta = pd.read_csv(img_meta_f)
        
    
    def __del__(self):
        pass
    
    def get_region_feat(self, i):
        idx1, idx2 = self.feat_indices[i, :]
        
        f1 = self.im_feat[idx1]
        f2 = self.im_feat[idx2]
        
        return f1, f2
        
    def get_example(self, i):
        # get region features
        r1, r2 = self.get_region_feat(i)
        
        # get phrase feature
        x1, x2 = self._get_entity(i)

        # get label
        y = self._get_label(i)
        y = np.asarray(y, dtype=np.int32)
        
        image = np.vstack([r1, r2])
        gt_r1 = 0
        gt_r2 = 1
        
        return image, x1, x2, gt_r1, gt_r2, y

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

def weighted_avrage_region(roi_feats, phr_feats, get_presoftmax=False):
    phr_feats = F.expand_dims(phr_feats, axis=1)
    
    # compute region weights
    weights = F.matmul(roi_feats, phr_feats, transb=True)
    softmax_weights = F.softmax(weightss, axis=1)
    
    roi_feats = roi_feats * F.broadcast_to(softmax_weights, roi_feats.shape)
    roi_feats = F.sum(roi_feats, axis=1)
    
    if get_presoftmax:
        return roi_feats, F.squeeze(weights)
    else:
        return roi_feats, F.squeeze(softmax_weights)
        
class PhraseGroundingNet(chainer.Chain):
    def __init__(self, attention_net):
        super(PhraseGroundingNet, self).__init__()
        with self.init_scope():
            self.attention_net = attention_net
    
    def __call__(self, region_feats, phr1_feats, phr2_feats, region_label1, region_label2, l):
        _, l_att1 = self.attention_net(phr1_feats, region_feats, region_label1)
        _, l_att2 = self.attention_net(phr2_feats, region_feats, region_label2)
        loss = (l_att1 + l_att2) * .5
        reporter.report({'loss': loss}, self)
        return loss

class FusionNet(chainer.Chain):
    def __init__(self):
        super(FusionNet, self).__init__()
        with self.init_scope():
            self.setup_layers()

    def setup_layers(self):
        h_size = 1000
        # fusenet for phrase and region features
        self.fuse_r1 = L.Linear(None, h_size, initialW=initializers.HeNormal())
        self.bn_1 = L.BatchNormalization(h_size)
        self.fuse_r2 = L.Linear(None, h_size, initialW=initializers.HeNormal())
        self.fuse_p = L.Linear(None, h_size, initialW=initializers.HeNormal(), nobias=True)
        self.bn_2 = L.BatchNormalization(h_size)
        self.fuse_2 = L.Linear(None, 300, initialW=initializers.HeNormal())
        self.bn_3 = L.BatchNormalization(300)

    def __call__(self, Xvis, Xp):
        Xvis = F.relu(self.bn_1(self.fuse_r1(Xvis)))
        h = F.relu(self.bn_2(self.fuse_r2(Xvis) + self.fuse_p(Xp)))
        h = F.relu(self.bn_3(self.fuse_2(h)))
        return h

class ProjectionNet(chainer.Chain):
    def __init__(self):
        super(ProjectionNet, self).__init__()
        with self.init_scope():
            self.setup_layers()

    def setup_layers(self):
        h_size = 1000
        # fusenet for phrase and region features
        self.fuse_p = L.Linear(None, h_size, initialW=initializers.HeNormal(), nobias=True)
        self.bn_2 = L.BatchNormalization(h_size)
        self.fuse_2 = L.Linear(None, 300, initialW=initializers.HeNormal())
        self.bn_3 = L.BatchNormalization(300)

    def __call__(self, Xp):
        h = F.relu(self.bn_2(self.fuse_p(Xp)))
        h = F.relu(self.bn_3(self.fuse_2(h)))
        return h

class ClassifierNet(chainer.Chain):
    def __init__(self, dr_ratio=.4):
        self._dr_ratio = dr_ratio
        super(ClassifierNet, self).__init__()
        with self.init_scope():
            self.mlp_1 = L.Linear(None, 128, initialW=initializers.HeNormal())
            self.mlp_2 = L.Linear(None, 128, initialW=initializers.HeNormal(), nobias=True)
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
        reporter.report({'loss':loss, 'precision': precision, 'recall': recall, 'f1': fbeta}, self)
        return loss


class AttentionNetGTP(chainer.Chain):
    def __init__(self, dr_ratio=.4):
        super(AttentionNetGTP, self).__init__()
        self._dr_ratio = dr_ratio
        h_size_att = 500
        with self.init_scope():
            # attention net
            self.fc_ar = L.Linear(None, h_size_att, initialW=initializers.LeCunNormal())
            self.fc_ap = L.Linear(None, h_size_att, initialW=initializers.LeCunNormal(), nobias=True)      
            self.fc_att = L.Linear(None, 1, initialW=initializers.HeNormal(), nobias=True)

    def __call__(self, phr_feats, region_feats, region_label=None):
        B, N, C = region_feats.shape

        hr = F.reshape(region_feats, (-1, C))

        hr =  self.fc_ap(hr)
        hr = F.reshape(hr, (B, N, -1))

        hp = self.fc_ar(phr_feats)
        hp = F.expand_dims(hp, axis=1)

        h = F.relu(F.broadcast_to(hp, hr.shape) + hr)
        h = F.reshape(h, (-1, h.shape[-1]))

        a = self.fc_att(F.dropout(h, self._dr_ratio))
        a = F.reshape(a, (B, N))

        # attention loss
        loss = None
        if region_label is not None:
            loss = F.softmax_cross_entropy(a, region_label)
            reporter.report({'loss': loss}, self)
        a = F.softmax(a, axis=1)
        return a, loss

class AttentionNetWTL(chainer.Chain):
    def __init__(self):
        super(AttentionNetWTL, self).__init__()
        h_size_att = 500
        with self.init_scope():            
            # attention net
            self.fc_ar = L.Linear(None, h_size_att, initialW=initializers.LeCunNormal())
            self.fc_ap = L.Linear(None, h_size_att, initialW=initializers.LeCunNormal())        

    def __call__(self, phr_feats, region_feats, region_label=None):
        B, N, C = region_feats.shape

        hr = F.reshape(region_feats, (-1, C))
        hr = self.fc_ar(hr)
        hr = F.reshape(hr, (B, N, -1))

        hp = self.fc_ap(phr_feats)
        hp = F.expand_dims(hp, axis=1)

        a = F.matmul(hr, hp, transb=True)
        a = F.reshape(a, (B, N))

        # attention loss
        loss = None
        if region_label is not None:
            loss = F.softmax_cross_entropy(a, region_label)

        a = F.softmax(a, axis=1)
        reporter.report({'loss': loss}, self)
        return a, loss   

class ParaphraseNet(chainer.Chain):
    def __init__(self, projection_net, classifier_net):
        super(ParaphraseNet, self).__init__()

        with self.init_scope():
            self.projection_net = projection_net
            self.classifier_net = classifier_net
    def predict(self, Xp1, Xp2, L):
        h1 = self.projection_net(Xp1)
        h2 = self.projection_net(Xp2)
        _ = self.classifier_net(h1, h2, L)
        return self.classifier_net.y, self.classifier_net.t
        
    def __call__(self, Xp1, Xp2, L):
        h1 = self.projection_net(Xp1)
        h2 = self.projection_net(Xp2)
        loss = self.classifier_net(h1, h2, L)
        reporter.report({'loss': loss}, self)
        return loss

class iParaphraseNet(chainer.Chain):
    def __init__(self, fusion_net, classifier_net, attention_net=None, kl_on=False, alpha=None):
        super(iParaphraseNet, self).__init__()
        self.use_gt_attention = (attention_net is None)

        with self.init_scope():
            self.fusion_net = fusion_net
            self.classifier_net = classifier_net
            if attention_net is not None:
                self.attention_net = attention_net
        
        self._kl_on = kl_on
        self.alpha = alpha

    def select_gt_region(self, region_feats, region_label):
        x = [F.embed_id(i, W) for i, W in zip(region_label[:, None], region_feats) ]
        return F.vstack(x)

    def compute_weigted_feat(self, region_feats, att):
        att = F.expand_dims(att, axis=-1)
        region_feats = region_feats * F.broadcast_to(att, region_feats.shape)
        return F.sum(region_feats, axis=1)

    def predict(self, Xr, Xp1, Xp2, Lr1, Lr2, L):
        if self.use_gt_attention:
            hr1 =self.select_gt_region(Xr, Lr1)
            hr2 =self.select_gt_region(Xr, Lr2)
            l_att = 0
        else:
            att1, l_att1 = self.attention_net(Xp1, Xr, Lr1)
            att2, l_att2 = self.attention_net(Xp2, Xr, Lr2)
            hr1 = self.compute_weigted_feat(Xr, att1)
            hr2 = self.compute_weigted_feat(Xr, att2)
            l_att = (l_att1 + l_att2) * .5
        
        h1 = self.fusion_net(hr1, Xp1)
        h2 = self.fusion_net(hr2, Xp2)

        _ = self.classifier_net(h1, h2, L)
        return self.classifier_net.y, self.classifier_net.t

    def __call__(self, Xr, Xp1, Xp2, Lr1, Lr2, L):
        if self.use_gt_attention:
            hr1 =self.select_gt_region(Xr, Lr1)
            hr2 =self.select_gt_region(Xr, Lr2)
            l_att = 0
        else:
            att1, l_att1 = self.attention_net(Xp1, Xr, Lr1)
            att2, l_att2 = self.attention_net(Xp2, Xr, Lr2)
            hr1 = self.compute_weigted_feat(Xr, att1)
            hr2 = self.compute_weigted_feat(Xr, att2)
            l_att = (l_att1 + l_att2) * .5
        
        h1 = self.fusion_net(hr1, Xp1)
        h2 = self.fusion_net(hr2, Xp2)

        l_cls = self.classifier_net(h1, h2, L)
        
        if self._kl_on:
            l_kl = kl_loss(att1, att2, L, alpha=self.alpha)
            loss = l_cls + l_att + l_kl
            reporter.report({'loss': loss, 'kl_loss': l_kl}, self)
        else:
            loss = l_cls + l_att
            reporter.report({'loss': loss}, self)

        return loss

class iOnlyNet(chainer.Chain):
    def __init__(self, projection_net, classifier_net, attention_net=None, kl_on=False, alpha=None):
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
        x = [F.embed_id(i, W) for i, W in zip(region_label[:, None], region_feats) ]
        return F.vstack(x)

    def compute_weigted_feat(self, region_feats, att):
        att = F.expand_dims(att, axis=-1)
        region_feats = region_feats * F.broadcast_to(att, region_feats.shape)
        return F.sum(region_feats, axis=1)

    def predict(self, Xr, Xp1, Xp2, Lr1, Lr2, L):
        if self.use_gt_attention:
            hr1 =self.select_gt_region(Xr, Lr1)
            hr2 =self.select_gt_region(Xr, Lr2)
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
            hr1 =self.select_gt_region(Xr, Lr1)
            hr2 =self.select_gt_region(Xr, Lr2)
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

def kl_loss(p1, p2, l, alpha=.5):
    eps=1e-5
    log_p1 = F.log(p1 + eps)
    log_p2 = F.log(p2 + eps)

    kl1 = F.sum(p1 * (log_p2 - log_p1), axis=1)
    kl2 = F.sum(p2 * (log_p1 - log_p2), axis=1)
    kl = - (kl1 + kl2) / 2
    kl_neg = F.relu(alpha - kl)
    loss = (l * kl + (1-l) * kl_neg) * .5

    return F.mean(loss)

def get_dataset(split='val', mode=None, v_feat='cca', p_feat='cca', skip=None, san_check=False, data_dir='./'):
    print(split)
    
    if mode == 'plclc-wo-vis':
        p_feat = 'wea' # dummy
    
    phrase_file = data_dir + 'data/phrase_pair_%s.csv'% split
    phrase_feature_file = data_dir + 'data/phrase_feat/%s/%s.npy'% (p_feat, split)
    img_feature_file = data_dir + 'data/region_feat/%s/%s.h5'% (v_feat, split)
    unique_phrase_file = data_dir + 'data/phrase_misc/%s_uniquePhrases' % split

    if mode in ['plclc-roi', 'plclc-wo-vis']:
        phrase_file = data_dir + 'data/pl-clc/phrase_pair_wt_plclcbbox_%s.csv' % split
        img_feature_file = [data_dir + 'data/region_feat/pl-clc-frcnn/%s.npy' % split,
                           data_dir + 'data/pl-clc/pl-clc-frcnn_indices_%s.npy' % split]
        
    elif mode == 'gt-roi':
        phrase_file = data_dir + 'data/gt-roi/phrase_pair_%s.csv'% split
        img_feature_file = [data_dir + 'data/region_feat/gt-roi-frcnn/%s.npy' % split,
                           data_dir + 'data/gt-roi/gt-roi-frcnn_indices_%s.npy' % split]

    if mode == 'wo-vis':
        dataset = EntityDatasetPhraseFeat(phrase_file, phrase_feature_file, unique_phrase_file,san_check=san_check)
        cvrt_fun = concat_examples
    elif mode in ['gt-roi', 'plclc-roi', 'plclc-wo-vis']:
        dataset = RegionEntityDatasetGT(
                    phrase_file,
                    phrase_feature_file,
                    unique_phrase_file,
                    img_feature_file,
                    san_check=san_check
                    )

    else:
        dataset = RegionEntityDatasetPhraseFeatwtROIWeights(
                    phrase_file,
                    phrase_feature_file,
                    unique_phrase_file,
                    img_feature_file,
                    san_check=san_check
                    )
        cvrt_fun = concat_examples

    return dataset, concat_examples

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

def setup_model(mode, dr_ratio=0.0, alpha=None):
    if mode in ['gt-roi', 'plclc-roi', 'plclc-wo-vis']:
        attention_net = None
        kl_on = False
    elif mode in ['gtp', 'gtp-kl', 'gtp-att']:
        attention_net = AttentionNetGTP(dr_ratio)
        kl_on = (mode == 'gtp-kl')
    elif mode in ['wtl', 'wtl-kl', 'wtl-att']:
        attention_net = AttentionNetWTL()
        kl_on = (mode == 'wtl-kl')
    elif mode == 'wo-vis':
        pass
    else:
        raise RuntimeError('invalid mode: %s' % mode)
    
    if mode in ['gtp-att', 'wtl-att']:
        model = PhraseGroundingNet(attention_net)
    else:
        classifier_net = ClassifierNet(dr_ratio)
        if mode == 'wo-vis':
            projection_net = ProjectionNet()
            model = ParaphraseNet(projection_net, classifier_net)
        elif mode == 'plclc-wo-vis':
            projection_net = ProjectionNet()
            model = iOnlyNet(projection_net,  classifier_net, attention_net, kl_on=kl_on)
        else:
            fusion_net = FusionNet()
            model = iParaphraseNet(fusion_net, classifier_net, attention_net, kl_on=kl_on)
    return model

def train(
    san_check = False,
    epoch = 5,
    lr = 0.001,
    lr_att = 0.001,
    dr_ratio = .4,
    b_size = 500,
    device = 0,
    w_decay = None,
    mode = 'gtp',
    intype = 'frcnn+wea',
    out_pref = './checkpoints/',
    # resume='',
    alpha=None
    ):
    args = locals()

    grounding_mode = False

    grounding_mode = (mode in ['gtp-att', 'wtl-att'])
    v_feat_type, p_feat_type = intype.split('+')

    out_base = out_pref+'%s-%s_'%(intype, mode)
    time_stamp = dt.now().strftime("%Y%m%d-%H%M%S")
    saveto = out_base + 'sc_' * san_check + time_stamp + '/'
    os.makedirs(saveto)
    json.dump(args, open(saveto+'args', 'w'))
    print('output to', saveto)
    print('setup dataset...')

    train, conv_f = get_dataset('train', mode=mode, v_feat=v_feat_type, p_feat=p_feat_type, san_check=san_check, data_dir='/home/mayu-ot/local_data/loc_iparaphrase/')
    val, _ = get_dataset('val', mode=mode, v_feat=v_feat_type, p_feat=p_feat_type, san_check=san_check, data_dir='/home/mayu-ot/local_data/loc_iparaphrase/')
    
    if chainer.config.multiproc:
        train_iter = MultiprocessIterator(train, b_size, n_processes=2)
        val_iter = MultiprocessIterator(val, b_size, shuffle=False, repeat=False, n_processes=2)
    else:
        train_iter = SerialIterator(train, b_size)
        val_iter = SerialIterator(val, b_size, shuffle=False, repeat=False)

    print('setup a model ...')
    model = setup_model(mode, dr_ratio, alpha)
    opt = chainer.optimizers.Adam(lr)
    opt.setup(model)

    # # set updaterules for attention net
    # for path, param in model.namedparams():
    #     if path.split('/')[1] == 'attention_net':
    #         param.update_rule.hyperparam.alpha = lr_att
    #         # param.update_rule = chainer.update_rules.MomentumSGD(lr_att).create_update_rule()

    if device:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    if w_decay:
        opt.add_hook(chainer.optimizer.WeightDecay(w_decay), 'hook_dec')

    updater = training.StandardUpdater(train_iter, opt, converter=concat_examples, device=device)
    trainer = training.Trainer(updater, (epoch, 'epoch'), saveto)

    val_interval = (1, 'epoch') if san_check else (500, 'iteration')
    log_interval = (1, 'iteration') if san_check else (10, 'iteration')
    plot_interval = (1, 'iteration') if san_check else (10, 'iteration')
    prog_interval = 1 if san_check else 10

    trainer.extend(extensions.Evaluator(val_iter, model, converter=concat_examples, device=device),
                trigger=val_interval)
    
    if not san_check:
        trainer.extend(extensions.ExponentialShift('alpha', 0.5), trigger=(1, 'epoch'))
    
    # # Comment out to enable visualization of a computational graph.
    # trainer.extend(extensions.dump_graph('main/loss'))
    if not san_check:
        ## Comment out next line to save a checkpoint at each epoch, which enable you to restart training loop from the saved point. Note that saving a checkpoint may cost a few minutes.
        trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
        if not grounding_mode:
            best_val_trigger = training.triggers.MaxValueTrigger('validation/main/classifier_net/f1', trigger=val_interval)
            trainer.extend(extensions.snapshot_object(
                model, 'model'), trigger=best_val_trigger)
        else:
            trainer.extend(extensions.snapshot_object(
                model, 'model'), trigger=training.triggers.MinValueTrigger('validation/main/loss', trigger=val_interval))
    trainer.extend(extensions.LogReport(trigger=log_interval, postprocess=postprocess))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    # trainer.extend(extensions.PrintReport([
    #     'epoch', 'iteration', 'main/loss', 'validation/main/loss','main/f1', 'validation/main/f1', 'lr'
    # ]), trigger=log_interval)

    if mode not in ['wo-vis', 'gtp-att', 'wtl-att', 'plclc-wo-vis']:
        logged_links = [
            model.fusion_net.fuse_p,
            model.fusion_net.fuse_r2
        ]
        statistics = {
            'min': cupy.min,
            'max': cupy.max,
        }
        trainer.extend(extensions.ParameterStatistics(logged_links, statistics, trigger=log_interval))

    plot_items = ['main/loss', 'validation/main/loss']

    trainer.extend(extensions.PlotReport(plot_items, file_name='loss.png', trigger=plot_interval))
    if not grounding_mode:
        trainer.extend(extensions.PlotReport(['main/classifier_net/f1', 'validation/main/classifier_net/f1'], file_name='f-measure.png', trigger=plot_interval))

    if (mode in ['wtl', 'gtp']) and (not grounding_mode):
        plot_items = ['main/attention_net/loss', 'validation/main/attention_net/loss']
        trainer.extend(extensions.PlotReport(plot_items, file_name='loss_att.png', trigger=plot_interval))
        plot_items = ['main/classifier_net/loss', 'validation/main/classifier_net/loss']
        trainer.extend(extensions.PlotReport(plot_items, file_name='loss_cls.png', trigger=plot_interval))
    
    # trainer.extend(extensions.ProgressBar(update_interval=prog_interval))

    # if resume:
    #     print('load terminated training', resume)
    #     chainer.serializers.load_npz(resume, trainer)

    print('start training')
    trainer.run()

    chainer.serializers.save_npz(saveto+'final_model', model)
    
    if not grounding_mode:
        return best_val_trigger._best_value

def get_prediction(model_dir, split, device=None):
    model_dir = model_dir+'/' if model_dir[-1] != '/' else model_dir

    settings = json.load(open(model_dir+'args'))

    mode = settings['mode']
    v_feat_type, p_feat_type = settings['intype'].split('+')
    
    print('setup a model ...')
    model = setup_model(mode, dr_ratio=0.0)
    chainer.serializers.load_npz(model_dir+'model', model)

    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    test, _ = get_dataset(split, mode=mode, v_feat=v_feat_type, p_feat=p_feat_type)
    test_iter = SerialIterator(test, batch_size=300, repeat=False, shuffle=False)
    conv_f = concat_examples
    
    s_i = 0
    e_i = 0
    pred = np.zeros((len(test),), dtype=np.float32)
    
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
    df['ypred'] = y_pred
    
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))

    with open(model_dir + 'res_%s_scores.txt'%split, 'w') as f:
        f.write('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))

    df.to_csv(model_dir+'res_%s.csv'%split)
    
    
'''
def evaluate(model_dir, split, device=None):
    chainer.config.train = False
    df = get_prediction(model_dir, split, device)

    y_true = df.ytrue
    y_pred = df.ypred
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))

    with open(model_dir + 'res_%s_scores.txt'%split, 'w') as f:
        f.write('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))

    df.to_csv(model_dir+'res_%s.csv'%split)
'''

# def evaluate(model_dir, split, device=None):
#     chainer.config.train = False

#     settings = json.load(open(os.path.join(model_dir, 'settings.json')))
    
#     mode = settings['mode']
#     v_feat_type, p_feat_type = settings['intype'].split('+')

#     val, _ = get_dataset('val', mode=mode, v_feat=v_feat_type, p_feat=p_feat_type)
#     test, _ = get_dataset('test', mode=mode, v_feat=v_feat_type, p_feat=p_feat_type)
    
#     print('setup a model ...')
#     if mode in ['wtl-gt', 'wtl']:
#         model = iParaphraseNetROISupWTL(use_gt_attention=(mode == 'wtl-gt'))
#     elif mode in ['gtp-gt', 'gtp']:
#         model = iParaphraseNetROISupGTP(use_gt_attention=(mode == 'gtp-gt'))
#     elif mode == 'wo-vis':
#         model = ParaphraseNet()
#     else:
#         raise RuntimeError('invalid mode')

#     print('load', model_dir)
#     chainer.serializers.load_npz(os.path.join(model_dir, 'model'), model)
     
#     if device is not None:
#         chainer.cuda.get_device_from_id(device).use()
#         model.to_gpu()

#     best_threshold = None
    
#     for dataset in [val, test]:
#         N = len(dataset)
#         b_size = 1000

#         gt = []
#         pred = []

#         print(N)

#         bar = ProgressBar()
#         with chainer.using_config('enable_backprop', False):
#             for i in bar(range(0, len(test), b_size)):
#                 batch = test[i:i+b_size]
#                 batch = concat_examples(batch, device=device)
                
#                 y = model.predict(*batch[:-1])
#                 y.to_cpu()
#                 pred.append( y.data )

#                 l = batch[-1]
#                 gt.append(chainer.cuda.to_cpu(l))
            
#         y_true = np.hstack(gt)
#         y_pred = np.hstack(pred)

#         if best_threshold is None:
#             precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
#             f1 = 2 * (precision * recall) / (precision + recall)
#             best_threshold = thresholds[f1.argmax()]
#             print('best threshold: %.2f, best f1: %.2f' % (best_threshold, f1.max() * 100))
#             continue
#         else:
#             y_pred = y_pred > best_threshold


#     prec = precision_score(y_true, y_pred)
#     rec = recall_score(y_true, y_pred)

#     f1 = f1_score(y_true, y_pred)
#     print('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))

#     with open(model_dir + 'res_%s_scores.txt'%split, 'w') as f:
#         f.write('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='training script for a paraphrase classifier')
    parser.add_argument('--lr', '-lr', type=float, default=0.01,
                        help='learning rate <float>')
    parser.add_argument('--lr_att', type=float, default=0.01,
                        help='weight for attention loss <float>')
    parser.add_argument('--device', '-d', type=int, default=None,
                        help='gpu device id <int>')
    parser.add_argument('--b_size', '-b', type=int, default=500,
                        help='minibatch size <int> (default 500)')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='maximum epoch <int>')
    parser.add_argument('--san_check', '-sc', action='store_true',
                        help='sanity check mode')
    parser.add_argument('--w_decay', '-wd', type=float, default=None,
                        help='weight decay <float>')
    parser.add_argument('--dr_ratio', '-dr', type=float, default=0.0,
                        help='dropout ratio <float>')
    parser.add_argument('--settings', type=str, default=None,
                        help='path to arg file')
    parser.add_argument('--eval', type=str, default=None,
                        help='path to an output directory <str>. the model will be evaluated.')
    parser.add_argument('--mode', '-m', type=str, default='wo-vis')
    parser.add_argument('--intype', type=str, default='cca+cca')
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

            prev_args = json.load(open(os.path.join(os.path.dirname(args.settings), 'args')))
            args_dic.update(prev_args)
            args_dic['settings'] = settings
            args_dic['epoch'] = epoch
            args_dic['device'] = device
            args_dic['out_pref'] = out_pref

        train(
            san_check = args_dic['san_check'],
            epoch = args_dic['epoch'],
            lr = args_dic['lr'],
            lr_att = args_dic['lr_att'],
            dr_ratio = args_dic['dr_ratio'],
            b_size = args_dic['b_size'],
            device = args_dic['device'],
            w_decay = args_dic['w_decay'],
            mode = args_dic['mode'],
            intype = args_dic['intype'],
            out_pref = args_dic['out_pref'])

if __name__ == '__main__':
    main()