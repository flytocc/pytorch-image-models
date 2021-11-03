import numpy as np

scores = np.load('scores.npy')
preds = np.load('preds.npy')
targets = np.load('targets.npy')

for c in range(1000):
    scores_per_category = scores[preds == c]
    scores_per_category.sort()
    gt = targets == c
    pr_list = []
    rc_list = []
    for s in scores_per_category:
        positive = (preds == c) & (scores >= s)
        TP = (gt & positive).sum()
        FP = (~gt & positive).sum()
        FN = (gt & ~positive).sum()
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        pr_list.append(precision)
        rc_list.append(recall)
    print()