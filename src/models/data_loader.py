import os
import numpy as np
import chainer
import pandas as pd
import imageio
from chainer.dataset.convert import to_device
from collections import defaultdict
import json
from nltk.metrics import edit_distance
import random


def lang_iou(x, y):
    x = set(x.split('+'))
    y = set(y.split('+'))
    inter = x.intersection(y)
    union = x.union(y)
    iou = len(inter) / len(union)
    return iou


def get_phrase_ious(df):
    p_ious = []
    for _, row in df.iterrows():
        p_iou = lang_iou(row.phrase1, row.phrase2)
        p_ious.append(p_iou)

    p_ious = np.asarray(p_ious)
    df['p_iou'] = p_ious
    return df


def get_agg_roi_df(split):
    gtroi_df = pd.read_csv(
        'data/phrase_localization/gt-roi/gt_roi_cord_%s.csv' % split, index_col=0)
    groups = gtroi_df.groupby(['image', 'org_phrase'])
    agg_df = groups.agg({
        'ymin': np.min,
        'xmin': np.min,
        'ymax': np.max,
        'xmax': np.max
    })
    return agg_df


class PhraseDataset(chainer.dataset.DatasetMixin):
    def __init__(self, split):
        df = pd.read_csv('data/%s.csv' % split,
                         usecols=['original_phrase1',
                                  'original_phrase2',
                                  'ytrue'])

        self.phrase_a = df.values[:, 0]
        self.phrase_b = df.values[:, 1]
        self.label = df.values[:, 2]

    def __len__(self):
        return len(self.phrase_a)

    def get_example(self, i):
        return self.phrase_a[i], self.phrase_b[i], self.label[i]


class BBoxDataset(chainer.dataset.DatasetMixin):
    def __init__(self, split):
        self.df = get_agg_roi_df(split)
        self.df['n_id'] = range(len(self.df))

    def __len__(self):
        return len(self.df)

    def get_phrase(self, i):
        return self.df.iloc[i].name[1]

    def get_image_id(self, i):
        return self.df.iloc[i].name[0]

    def get_example(self, i):
        row = self.df.iloc[i]
        im_id, _ = row.name
        im = imageio.imread('data/flickr30k-images/%i.jpg' % im_id)
        im = np.asarray(im)
        gt_roi = row[['ymin', 'xmin', 'ymax', 'xmax']]
        gt_roi = np.asarray(gt_roi).astype('f')
        return im, gt_roi


def prepare_vis_feat_indices(bbox_df, pair_df):
    phr1 = pair_df[['image', 'original_phrase1']].drop_duplicates()
    phr1 = phr1.rename(index=str, columns={'original_phrase1': 'phrase'})
    phr2 = pair_df[['image', 'original_phrase2']].drop_duplicates()
    phr2 = phr2.rename(index=str, columns={'original_phrase2': 'phrase'})
    d_df = pd.concat([phr1, phr2]).drop_duplicates()

    indices = {}
    cur_img = 0
    for _, row in d_df.iterrows():
        img, phrase = row
        if img != cur_img:
            rows = bbox_df[bbox_df.image == img]
            cur_img = img
            indices.setdefault(str(img), {})

        dist = list(
            map(lambda x: edit_distance(phrase.lower(), x), rows.phrase))
        row = rows.iloc[np.argmin(dist)]
        index = row.name
        indices[str(img)][phrase] = int(index)

    return indices


class DDPNBBoxDataset(chainer.dataset.DatasetMixin):
    def __init__(self, split):
        df = pd.read_csv('data/phrase_localization/ddpn/fix_split_%s.csv' % split)
        sub_df = pd.DataFrame(
            df.iloc[:, 5:9].values, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        sub_df['image'] = df.image
        sub_df['phrase'] = df.phrase
        self.df = sub_df

    def __len__(self):
        return len(self.df)

    def get_phrase(self, i):
        return self.df.iloc[i].phrase

    def get_image_id(self, i):
        return self.df.iloc[i].image

    def get_example(self, i):
        row = self.df.iloc[i]
        im_id = row.image
        im = imageio.imread('data/flickr30k-images/%i.jpg' % im_id)
        im = np.asarray(im)

        bbox = row[['ymin', 'xmin', 'ymax', 'xmax']]
        bbox = np.asarray(bbox).astype('f')

        return im, bbox


class PLCLCBBoxDataset(DDPNBBoxDataset):
    def __init__(self, split):
        df = pd.read_csv('data/phrase_localization/plclc/localization_%s.csv' % split, index_col=0)
        pair_df = pd.read_csv('data/phrase_pair_%s.csv' % split, index_col=0)
        df = df.drop_duplicates(['image', 'org_phrase'])
        df = df.reset_index()
        df = df.rename(columns={'org_phrase': 'phrase'})
        df = df[['image', 'phrase', 'xmin', 'ymin', 'xmax', 'ymax']]
        self.df = df


def downsample_easynegatives(df):
    p_ious = df.p_iou
    easy_pos, = np.where(np.logical_and(df.ytrue == True, p_ious == 0))
    easy_neg, = np.where(np.logical_and(df.ytrue == False, p_ious == 0))

    # random.seed(1234)
    drop_n = len(easy_neg)-len(easy_pos)
    drop_items = random.sample(easy_neg.tolist(), drop_n // 2)
    return df.drop(drop_items)


class iParaphraseDataset(chainer.dataset.DatasetMixin):
    def __init__(self, split, san_check=False):

        if split == 'train':
            self.resample()
            pair_data = self.pair_data
        else:
            pair_data = pd.read_csv('data/phrase_pair_%s.csv' % split, index_col=0)

        if san_check:
            skip = 2000 if split == 'train' else 1000
            pair_data = pair_data.iloc[::skip]
            pair_data = pair_data.reset_index(drop=True)

        self.pair_data = pair_data

        self.gtroi_data = pd.read_csv(
            'data/phrase_localization/gt-roi/gt_roi_cord_%s.csv' % split, index_col=0)
        self.img_root = 'data/flickr30k-images/'

        # get phrse indices
        p2i_dict = defaultdict(lambda: -1)
        with open('data/language/%s_uniquePhrases' % split) as f:
            for i, line in enumerate(f):
                p2i_dict[line.rstrip()] = i

        self._p2i_dict = p2i_dict
        self._feat = np.load('data/language/phrase_feat/wea/%s.npy' % split)

        print('%s data: %i pairs' % (split, len(self.pair_data)))

    def __len__(self):
        return len(self.pair_data)

    def resample(self):
        print('resample dataset ...\n')
        fname = 'data/phrase_pair_train_wt_pious.csv'

        if os.path.exists(fname):
            pair_data = pd.read_csv(fname, index_col=0)
        else:
            pair_data = pd.read_csv('data/phrase_pair_train.csv', index_col=0)
            pair_data = get_phrase_ious(pair_data)
            pair_data.to_csv(fname)

        pair_data = downsample_easynegatives(pair_data)
        pair_data = pair_data.reset_index(drop=True)

        self.pair_data = pair_data

    def get_phrases(self, i):
        return self.pair_data.at[i, 'phrase1'], self.pair_data.at[i, 'phrase2']

    def get_phrase_feat(self, i):
        phr1, phr2 = self.get_phrases(i)
        x1 = self._feat[self._p2i_dict[phr1]]
        x2 = self._feat[self._p2i_dict[phr2]]
        return x1, x2

    def read_image(self, i):
        img_id = self.pair_data.at[i, 'image']
        img = imageio.imread(os.path.join(self.img_root, str(img_id)) + '.jpg')
        return img

    def get_example(self, i):
        img = self.read_image(i)
        phr_1, phr_2 = self.get_phrase_feat(i)
        gt_roi_1, gt_roi_2 = self.get_gt_roi(i)
        l = self.pair_data.at[i, 'ytrue']
        return img, phr_1, phr_2, gt_roi_1, gt_roi_2, l

    def get_gt_roi(self, i):
        img_id = self.pair_data.at[i, 'image']
        phr_1 = self.pair_data.at[i, 'phrase1']
        phr_2 = self.pair_data.at[i, 'phrase2']
        rois = self.gtroi_data[self.gtroi_data.image == img_id]

        gt_rois = []
        for phr in [phr_1, phr_2]:
            roi = rois[rois.phrase == phr][['ymin', 'xmin', 'ymax', 'xmax']]
            gt_roi_min = roi.min(axis=0)
            gt_roi_max = roi.max(axis=0)
            roi = np.hstack((gt_roi_min[:2], gt_roi_max[-2:]))
            gt_rois.append(roi)

        return gt_rois[0], gt_rois[1]


class PreCompFeatDataset(iParaphraseDataset):
    def __init__(self, split, san_check=False):
        super(PreCompFeatDataset, self).__init__(split, san_check)
        self.setup(split)

    def setup(self, split):
        '''
        prepare
        self.bbox_df
        self.vis_feat
        '''
        raise NotImplementedError

    def get_feat_row_id(self, i):
        img_id = self.pair_data.at[i, 'image']
        phr1 = self.pair_data.at[i, 'original_phrase1']
        phr2 = self.pair_data.at[i, 'original_phrase2']
        nid_1 = self.bbox_df.at[(img_id, phr1), 'n_id']
        nid_2 = self.bbox_df.at[(img_id, phr2), 'n_id']

        if isinstance(nid_1, np.ndarray):
            nid_1 = nid_1[0]

        if isinstance(nid_2, np.ndarray):
            nid_2 = nid_2[0]

        return nid_1, nid_2

    def lget_bbox(self, img_id, phr):
        ymin = self.bbox_df.at[(img_id, phr), 'ymin']
        xmin = self.bbox_df.at[(img_id, phr), 'xmin']
        ymax = self.bbox_df.at[(img_id, phr), 'ymax']
        xmax = self.bbox_df.at[(img_id, phr), 'xmax']
        return ymin, xmin, ymax, xmax

    def get_bbox(self, i):
        img_id = self.pair_data.at[i, 'image']
        phr1 = self.pair_data.at[i, 'original_phrase1']
        phr2 = self.pair_data.at[i, 'original_phrase2']
        bbox_1 = self.lget_bbox(img_id, phr1)
        bbox_2 = self.lget_bbox(img_id, phr2)
        return bbox_1, bbox_2

    def get_vis_feat(self, i):
        nid_1, nid_2 = self.get_feat_row_id(i)
        return self.vis_feat[nid_1], self.vis_feat[nid_2]

    def get_jbbox(self, i):
        nid_1, nid_2 = self.get_feat_row_id(i)
        return self.jbbox[nid_1], self.jbbox[nid_2]

    def get_example(self, i):
        xvis_1, xvis_2 = self.get_vis_feat(i)
        phr_1, phr_2 = self.get_phrase_feat(i)
        l = self.pair_data.at[i, 'ytrue']

        return phr_1, phr_2, xvis_1, xvis_2, l


class GTJitterDataset(PreCompFeatDataset):
    def setup(self, split):
        self.bbox_df = BBoxDataset(split).df
        self.vis_feat = np.load(
            'data/phrase_localization/region_feat/jitter_roi-frcnn_asp0.66_off0.4/%s.npy' % split)
        if split == 'test':
            self.jbbox = np.load(
                'data/phrase_localization/region_feat/jitter_roi-frcnn_asp0.66_off0.4/rois_%s.npy'
                % split)


def get_most_similar(q, targ):
    best_d = np.inf
    for x in targ:
        d = edit_distance(q, x)
        if d < best_d:
            best_d = d
            res = x
        if best_d == 0:
            break
    return res


class DDPNDataset(PreCompFeatDataset):
    def setup(self, split):
        self.bbox_df = DDPNBBoxDataset(split).df
        self.vis_feat = np.load(
            'data/phrase_localization/region_feat/ddpn_roi-frcnn/%s.npy' % split)

        indices_file = 'data/phrase_localization/ddpn/vis_indices_%s.json' % split
        if os.path.exists(indices_file):
            vis_indices = json.load(open(indices_file))
        else:
            vis_indices = prepare_vis_feat_indices(self.bbox_df, self.pair_data)
            json.dump(vis_indices, open(indices_file, 'w'))

        self.vis_indices = vis_indices
        print('done')

    def __len__(self):
        return len(self.pair_data)

    def get_nid(self, img_id, phrase):
        map_dict = self.vis_indices[str(img_id)]
        try:
            return map_dict[phrase.lower()]
        except:
            r = get_most_similar(phrase.lower(), map_dict.keys())
            self.vis_indices[str(img_id)][phrase.lower()] = map_dict[r]  # add item
            return map_dict[r]

    def get_example(self, i):
        img_id = self.pair_data.at[i, 'image']
        phr_1 = self.pair_data.at[i, 'original_phrase1']
        phr_2 = self.pair_data.at[i, 'original_phrase2']

        nid_1 = self.get_nid(img_id, phr_1)
        nid_2 = self.get_nid(img_id, phr_2)

        xvis_1 = self.vis_feat[nid_1]
        xvis_2 = self.vis_feat[nid_2]

        phr_1 = self.pair_data.at[i, 'phrase1']
        phr_2 = self.pair_data.at[i, 'phrase2']
        phr_1 = self._feat[self._p2i_dict[phr_1]]
        phr_2 = self._feat[self._p2i_dict[phr_2]]

        l = self.pair_data.at[i, 'ytrue']

        return phr_1, phr_2, xvis_1, xvis_2, l


class PLCLCDataset(DDPNDataset):
    def setup(self, split):
        self.bbox_df = PLCLCBBoxDataset(split).df
        self.vis_feat = np.load(
            'data/phrase_localization/region_feat/plclc_roi-frcnn/%s.npy' % split)

        indices_file = 'data/phrase_localization/plclc/vis_indices_%s.json' % split
        if os.path.exists(indices_file):
            vis_indices = json.load(open(indices_file))
        else:
            vis_indices = prepare_vis_feat_indices(self.bbox_df,
                                                   self.pair_data)
            with open(indices_file, 'w') as f:
                json.dump(vis_indices, f)

        self.vis_indices = vis_indices

    def get_nid(self, img_id, phrase):
        map_dict = self.vis_indices[str(img_id)]
        try:
            return map_dict[phrase]
        except:
            r = get_most_similar(phrase, map_dict.keys())
            self.vis_indices[str(img_id)][phrase] = map_dict[r]  # add item
            return map_dict[r]
