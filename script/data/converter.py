# -*- coding: utf-8 -*-

import os
import json
import parse
import pandas as pd
import re
from itertools import combinations, product
from more_itertools import flatten
from collections import defaultdict
import progressbar


def save_phr2phr_score(in_file, out_file):
    image = []
    phrase1 = []
    phrase2 = []
    region1 = []
    region2 = []
    sentence1 = []
    sentence2 = []
    score = []

    tmpl_im_id = parse.compile("image_id: {:d}")
    tmpl_sent_id = parse.compile("sentence pair id: {:d}-{:d}")
    tmpl_item_1 = parse.compile("{:d}/{} : {} # {:d}/{} : {} # 0")
    tmpl_item_2 = parse.compile(
        "{:d}/{} : {} # {:d}/{} : {} # {:g} {:g} {:g} {:g}")

    with open(in_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line[0] == 'i':
                img, = tmpl_im_id.parse(line)
            elif line[0] == 's':
                sent1, sent2 = tmpl_sent_id.parse(line)
            else:
                if line.split('# ')[-1] == '0':
                    reg1, cat1, phr1, reg2, cat2, phr2 = tmpl_item_1.parse(
                        line)
                    s = 0
                else:
                    reg1, cat1, phr1, reg2, cat2, phr2, s1, s2, s3, s4 = tmpl_item_2.parse(
                        line)
                    s = s1 * s3

                # phr1 = phr1.replace('\'', '').replace('\"', '')
                # phr2 = phr2.replace('\'', '').replace('\"', '')
                image.append(img)
                phrase1.append(phr1)
                phrase2.append(phr2)
                region1.append(reg1)
                region2.append(reg2)
                sentence1.append(sent1)
                sentence2.append(sent2)
                score.append(s)

    df = pd.DataFrame({
        'image': image,
        'phrase1': phrase1,
        'phrase2': phrase2,
        'region1': region1,
        'region2': region2,
        'sentence1': sentence1,
        'sentence2': sentence2,
        'score': score
    })

    df.to_csv(out_file)


def convert_phr2phr_score_file(phrase_pair_file, phrase2phrase_score_file,
                               output_file):
    phrase_pair = pd.read_csv(phrase_pair_file)
    score_df = pd.read_csv(phrase2phrase_score_file)

    score = []
    for _, row in phrase_pair.iterrows():
        org_phrase1 = row['original_phrase1'].lower()
        org_phrase2 = row['original_phrase2'].lower()

        res = score_df[(score_df.image == row['image'])
                       & (score_df.phrase1 == org_phrase1) &
                       (score_df.phrase2 == org_phrase2)]
        if len(res) == 0:
            print('%i: %20s | %20s' % (row['image'], org_phrase1, org_phrase2))
            score.append(-1)
            continue
        score.append(res.score.values[0])

    df = phrase_pair.copy()
    df['trans_score'] = score
    df.to_csv(output_file)


def load_sentence(txt_file):
    sents = []
    with open(txt_file, 'r') as f:
        for i, line in enumerate(f):
            entities = []
            for item in re.findall(r'\[.*?\]', line):
                phr_id, category, phrase = parse.parse('[/EN#{:d}/{} {}]',
                                                       item)
                category = category.split('/')[0]
                if category != 'notvisual':
                    entities.append({
                        'cap_i': i,
                        'id': phr_id,
                        'category': category,
                        'org_phrase': phrase
                    })
            sents.append(entities)
    return sents


def load_preprocessed(line):
    sent_id = int(line[4])
    items = line[7:].split(' ## ')[:-1]

    items = [x.split() for x in items]
    items = [x if len(x) > 1 else x + ['<discarded-phrase>'] for x in items]

    entities = {}
    for eid_group, p in items:
        entities.setdefault(eid_group, []).append(p)

    return entities


def save_phrase_pairs(phr_file, sentence_dir, out_file):
    with open(phr_file, 'r') as f:
        items = []
        for line in f:
            if line[:5] == 'image':
                im_id = line.split()[-1]
                prepro_sentences = []
            elif line[:3] == 'sen':
                entities = load_preprocessed(line)
                prepro_sentences.append(entities)
            elif line == '\n':
                print(im_id)
                # load original entities
                sentences = load_sentence(
                    os.path.join(sentence_dir, im_id + '.txt'))

                for org_s, prepro_s in zip(sentences, prepro_sentences):
                    for entity in org_s:
                        id_category_key = '%i/%s' % (entity['id'],
                                                     entity['category'])
                        if prepro_s is None:
                            entity['prepro_phrase'] = '<discarded-phrase>'
                        elif id_category_key in prepro_s.keys():
                            entity['prepro_phrase'] = prepro_s[
                                id_category_key].pop(0)
                        else:
                            entity['prepro_phrase'] = '<discarded-phrase>'

                for s1, s2 in combinations(sentences, 2):
                    for ent1, ent2 in product(s1, s2):
                        if ent1['org_phrase'] == ent2['org_phrase']:
                            continue
                        y_true = (ent1['id'] == ent2['id']) and (
                            ent1['category'] == ent2['category'])
                        items.append([
                            im_id,
                            ent1['org_phrase'] + '/' + ent1['prepro_phrase'],
                            ent2['org_phrase'] + '/' + ent2['prepro_phrase'],
                            y_true
                        ])
    df = pd.DataFrame(items, columns=['image', 'phrase1', 'phrase2', 'ytrue'])
    df.to_csv(out_file)


def remove_trivial_match(in_file, out_file):
    df = pd.read_csv(in_file)

    item = []

    for _, row in df.iterrows():
        org_phr1, phr1 = row['phrase1'].split('/')
        org_phr2, phr2 = row['phrase2'].split('/')

        if phr1 == phr2:
            continue

        if (phr1 == '<discarded-phrase>') or (phr2 == '<discarded-phrase>'):
            continue

        item.append(
            [row['image'], phr1, phr2, org_phr1, org_phr2, row['ytrue']])

    df = pd.DataFrame(
        item,
        columns=[
            'image', 'phrase1', 'phrase2', 'original_phrase1',
            'original_phrase2', 'ytrue'
        ])
    df.to_csv(out_file)


def save_phrase2phrase_score(phrase_pair_file, uni_phrase_file, feat_file,
                             out_file):
    with open(uni_phrase_file, 'r') as f:
        all_phrase = f.read()
        all_phrase = all_phrase.split()

    phrase_dict = {k: i for i, k in enumerate(all_phrase)}

    # load phrase feature
    Xp = np.load(feat_file)

    df = pd.read_csv(phrase_pair_file)

    score = []
    for _, row in df.iterrows():
        phrase1 = row['phrase1']
        phrase2 = row['phrase2']
        score.append(
            (Xp[phrase_dict[phrase1]] * Xp[phrase_dict[phrase2]]).sum())

    df['score'] = pd.Series(score)
    df.to_csv(out_file)


def parse_entity_region_score(in_file, phrase_pair_file, out_name):
    df = pd.read_csv(phrase_pair_file)
    items = []

    # except_dict = defaultdict(int)
    # except_dict.update({"they ’re truck": "they’re truck", "Free B. . .": "Free B.. .", "the ' thumbs up '": "the 'thumbs up '"})

    with open(in_file, 'r') as f:
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

        for line_i, line in enumerate(f):

            if line[:3] == 'img':
                im_id, s_id, ent_id, group, phrase = parse.parse(
                    'img_{:d}:sen_{:d}:entity_{:d}/{}:{} ', line)

                # phrase = phrase.replace(' ’', "’")
                # if except_dict[phrase]:
                #     phrase = except_dict[phrase]

                # get preprocessed phrase
                sub_df = df[df.image == im_id]
                if any(sub_df.original_phrase1 == phrase):
                    cvrt_phrase = sub_df[sub_df.original_phrase1 ==
                                         phrase].phrase1.values[0]
                elif any(sub_df.original_phrase2 == phrase):
                    cvrt_phrase = sub_df[sub_df.original_phrase2 ==
                                         phrase].phrase2.values[0]
                else:
                    cvrt_phrase = 'None'
                BB = []
            elif line[:3] == 'box':
                box_id, score, xmin, ymin, xmax, ymax = parse.parse(
                    'box: {:d}, score: {:f}, coordinate: {:f}, {:f}, {:f}, {:f}, ',
                    line)
                items.append([
                    im_id, s_id, ent_id, group, phrase, cvrt_phrase, box_id,
                    xmin, ymin, xmax, ymax, score
                ])
            else:
                pass

            bar.update(line_i)

    df = pd.DataFrame(
        items,
        columns=[
            'image', 'sentence', 'entity', 'group', 'original_phrase',
            'phrase', 'bbox', 'xmin', 'ymin', 'xmax', 'ymax', 'score'
        ])
    df.to_csv(out_name)


def save_phrase_graph(in_file, out_dir):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    df = pd.read_csv(in_file)
    images = pd.unique(df.image)

    for im in images:
        sub_df = df[df.image == im]
        G = nx.Graph()
        G.add_weighted_edges_from([(row['phrase1'], row['phrase2'],
                                    row['score'])
                                   for _, row in sub_df.iterrows()])

        p2i = {p: i for i, p in enumerate(G.nodes())}
        edges = [([p2i[p1], p2i[p2]]) for p1, p2 in G.edges()]
        weights = [w for _, _, w in G.edges(data='weight')]

        graph_data = {
            'edges': edges,
            'weights': weights,
            'phrases': [p for p in G.nodes()]
        }
        json.dump(graph_data, open(os.path.join(out_dir, '%i.json' % im), 'w'))


def save_phrase2region_table(in_file, graph_dir, out_dir):
    print(in_file)
    print(graph_dir)
    df = pd.read_csv(in_file)
    images = pd.unique(df.image)

    for im in images:
        sub_df = df[df.image == im]
        Nr = sub_df.bbox.values.max()

        phrases = json.load(open(graph_dir + '%i.json' % im))['phrases']
        Mpr = np.zeros((len(phrases), Nr))

        for i, p in enumerate(phrases):
            #select first phrase-region score data
            if sub_df[sub_df.phrase == p].sentence.values.size == 0:
                print(im, p)
                raise RuntimeError
            sent_id = sub_df[sub_df.phrase == p].sentence.values.min()
            for _, row in sub_df[(sub_df.phrase == p)
                                 & (sub_df.sentence == sent_id)].iterrows():
                Mpr[i, row['bbox'] - 1] = row['score']

        np.save(os.path.join(out_dir, str(im)), Mpr)


def get_cvrt_dict(df):
    cvrt_dict = defaultdict(lambda: [])
    for _, row in df.iterrows():
        if row['phrase1'] not in cvrt_dict[row['original_phrase1']]:
            cvrt_dict[row['original_phrase1']].append(row['phrase1'])
        if row['phrase2'] not in cvrt_dict[row['original_phrase2']]:
            cvrt_dict[row['original_phrase2']].append(row['phrase2'])
    return cvrt_dict


def write_gt_cluster_label_file(in_file, out_file):
    '''
    infile: output if remove_trivial_match
    '''
    df = pd.read_csv(in_file)
    cvrt_dict = get_cvrt_dict(df)
    items = []
    sent_root = '/home/mayu-ot/Data/Dataset/Flickr30kEntities/Sentences/'
    for im_id in pd.unique(df.image.values):
        sub_df = df[df.image == im_id]
        sent = load_sentence(sent_root + '%i.txt' % im_id)
        phr_dict = {}
        for s in flatten(sent):
            phr_list = cvrt_dict[s['org_phrase']]  # list of converted phrases

            if len(phr_list) == 0:
                continue

            for phr in phr_list:
                if sub_df.phrase1.isin([phr]).any() or sub_df.phrase2.isin(
                    [phr]).any():
                    label = '%s/%s' % (s['id'], s['category'])
                    phr_dict[phr] = label
                    continue

        for k, v in phr_dict.items():
            items.append((im_id, k, v))

    out_df = pd.DataFrame(items, columns=['image', 'phrase', 'label'])
    out_df.to_csv(out_file)


def main():
    split = 'val'
    feat_type = 'fv+cca'
    out_base = 'data/pl-clc_cca/convert/'

    phrase_pair = out_base + 'phrase_pair_%s.csv' % split

    save_phrase_pairs(
        'data/pl-clc_cca/entity/%s/phraseWords' % split,
        '/home/mayu-ot/Data/Dataset/Flickr30kEntities/Sentences/',
        out_base + 'phrase_pair_%s.csv' % split)

    phrase_pair_filtered = out_base + 'phrase_pair_remove_trivial_match_%s.csv' % split

    remove_trivial_match(phrase_pair, phrase_pair_filtered)

    phrase_score_file = out_base + 'phrase_score_%s_%s.csv' % (feat_type,
                                                               split)

    save_phrase2phrase_score(
        phrase_pair_filtered,
        'data/pl-clc_cca/entity/%s/uniquePhrases' % split,
        'data/pl-clc_cca/entity/%s/textFeats_%s.npy' % (split, feat_type),
        phrase_score_file)

    p2r_score_file = out_base + 'entity_region_scores_%s.csv' % split
    parse_entity_region_score(
        'data/pl-clc_cca/entity_region_scores.%sSplit.txt' % split,
        phrase_pair_filtered, p2r_score_file)

    graph_dir = out_base + 'phrase_graph/%s/%s/' % (feat_type, split)
    save_phrase_graph(phrase_score_file, graph_dir)

    out_dir = 'data/pl-clc_cca/convert/phrase_region_score/cca+/%s/' % split
    save_phrase2region_table(p2r_score_file, graph_dir, out_dir)


if __name__ == '__main__':
    main()
