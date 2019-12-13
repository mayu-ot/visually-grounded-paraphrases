from itertools import combinations
from scipy.spatial.distance import squareform
from sklearn.cluster import AffinityPropagation
from tqdm import tqdm
import numpy as np
from itertools import combinations
import pandas as pd
from sklearn.metrics import adjusted_rand_score

def group_phrases(A):
    A = A.copy()
    groups = []
    i = 0
    while(i<len(A)):
        g = np.where(A[i]==1)[0]
        A[:, g] = -1
        groups.append(g)
        cand = np.where(A[i]==0)[0]
        if len(cand)==0:
            break
        i = cand[cand>i][0]
        
    label = np.zeros((len(A),), dtype=np.uint8)
    for i, g in enumerate(groups):
        label[g] = i
    return label

def eval(df, clutter=.5, damping=.5):
    ad_rands = []

    for img_id in tqdm(df.image.unique()):
        sub_df = df[df.image==img_id]
        phrases = pd.unique(sub_df[['phrase1', 'phrase2']].values.ravel())
        n = len(phrases)
        scores = []
        t_label = []

        for i_1, i_2 in combinations(range(n), 2):
            p1 = phrases[i_1]
            p2 = phrases[i_2]
            rows = sub_df[(sub_df.phrase1==p1)&(sub_df.phrase2==p2)]
            s = rows.pred.values[0] if len(rows) else 0
            t = rows.label.values[0] if len(rows) else 0
            scores.append(s)
            t_label.append(t)

        A = squareform(scores)
        T = squareform(t_label) + np.eye(n)

        pref = np.percentile(scores, clutter)
        af = AffinityPropagation(preference=pref, affinity='precomputed',damping=damping)
        label = af.fit_predict(A)
        gt_label = group_phrases(T)
        adr_s = adjusted_rand_score(gt_label, label)
        ad_rands.append(adr_s)
    
    return np.mean(ad_rands)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', type=str)
    args = parser.parse_args()
    
    df = pd.read_csv(args.pred_file)
    score = eval(df)
    print(score)