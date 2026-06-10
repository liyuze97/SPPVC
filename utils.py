import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def to_numpy(x):
    return x.cpu().numpy()


def to_tensor(x, device='cuda'):
    return torch.from_numpy(x.astype(np.float32)).to(device)


def evaluation(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    return acc, nmi, ari


def eval_aligned_detail(P_pred, index_mis_aligned, labels_0, labels_1):
    if torch.is_tensor(P_pred):
        P_pred = P_pred.cpu().numpy()

    idx_gt = np.array(index_mis_aligned)
    idx_pred = (P_pred @ index_mis_aligned).astype(int)

    label_match = (labels_0[idx_gt] == labels_1[idx_pred])

    return np.sum(label_match)