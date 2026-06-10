import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import torch
import random
from construct_graph import get_graph


def load_data(dataset, config):
    data_dir = "../Datasets/{}.mat".format(dataset)
    data, label, A = [], [], []
    mat = sio.loadmat(data_dir)

    if dataset == 'HandWritten':
        raw_X1 = mat['X'][0][0]
        raw_X2 = mat['X'][0][2]

        X1 = StandardScaler().fit_transform(raw_X1)
        X2 = StandardScaler().fit_transform(raw_X2)

        data.append(X1)
        data.append(X2)

        A.append(get_graph(X1, k=config['k'], method='euclidean'))
        A.append(get_graph(X2, k=config['k'], method='euclidean'))

        label = np.squeeze(mat['Y'].astype('int'))

    elif dataset == 'Caltech101-7':
        raw_X1 = mat['X'][0][3]
        raw_X2 = mat['X'][0][4]

        X1 = StandardScaler().fit_transform(raw_X1)
        X2 = StandardScaler().fit_transform(raw_X2)

        data.append(X1)
        data.append(X2)

        A.append(get_graph(X1, k=config['k'], method='pearson'))
        A.append(get_graph(X2, k=config['k'], method='pearson'))

        label = np.squeeze(mat['Y'].astype('int'))

    elif dataset == 'BDGP':
        raw_X1 = mat['X1']
        raw_X2 = mat['X2']

        X1 = StandardScaler().fit_transform(raw_X1)
        X2 = StandardScaler().fit_transform(raw_X2)

        data.append(X1)
        data.append(X2)

        A.append(get_graph(X1, k=config['k'], method='cosine'))
        A.append(get_graph(X2, k=config['k'], method='cosine'))

        label = np.squeeze(mat['Y'][0]).T

    elif dataset == 'WebKB':
        raw_X1 = mat['x1']
        raw_X2 = mat['x2']

        X1 = StandardScaler().fit_transform(raw_X1)
        X2 = StandardScaler().fit_transform(raw_X2)

        data.append(X1)
        data.append(X2)

        A.append(get_graph(X1, k=config['k'], method='pearson'))
        A.append(get_graph(X2, k=config['k'], method='pearson'))

        label = np.squeeze(mat['y'].astype('int'))

    elif dataset == 'Reuters':
        X1 = mat['x_train'][1]
        X2 = mat['x_train'][3]

        data.append(X1)
        data.append(X2)

        A.append(get_graph(X1, k=config['k'], method='cosine'))
        A.append(get_graph(X2, k=config['k'], method='cosine'))

        label = np.squeeze(mat['y_train'].astype('int'))

    elif dataset in ['Fashion']:
        X1 = mat['X1'].reshape(10000, -1).astype('float32')
        X2 = mat['X2'].reshape(10000, -1).astype('float32')

        data.append(X1)
        data.append(X2)

        A.append(get_graph(X1, k=config['k'], method='cosine'))
        A.append(get_graph(X2, k=config['k'], method='cosine'))
        label = np.squeeze(mat['Y'])

    return data, label, A


def aligned_data_split(n_all, test_prop, seed):
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil((1 - test_prop) * n_all).astype(int)
    train_idx = np.array(sorted(random_idx[0:train_num]))
    test_num = np.floor(test_prop * n_all).astype(int)
    test_idx = np.array(sorted(random_idx[-test_num:]))
    return train_idx, test_idx


def prepare_aligned_views(data, labels, A, aligned_ratio, auxiliary_view=1, device='cuda'):
    N = len(labels)

    aligned_indices = []
    unaligned_indices = []
    total_aligned = int(N * aligned_ratio)

    class_ids = np.unique(labels)
    for i, class_id in enumerate(class_ids):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)

        if i == len(class_ids) - 1:
            num_aligned = total_aligned - len(aligned_indices)
        else:
            num_aligned = int(len(class_indices) * aligned_ratio)

        aligned_indices.extend(class_indices[:num_aligned])
        unaligned_indices.extend(class_indices[num_aligned:])

    aligned_indices = np.sort(aligned_indices)
    unaligned_indices = np.sort(unaligned_indices)

    labels_0 = labels
    labels_1 = labels
    if aligned_ratio != 1:
        P_index = np.arange(N)

        shuffle_idx = np.random.permutation(unaligned_indices)
        P_index[unaligned_indices] = shuffle_idx

        data[auxiliary_view] = data[auxiliary_view][P_index]
        A[auxiliary_view] = A[auxiliary_view][P_index][:, P_index]
        labels_0 = labels
        labels_1 = labels[P_index]

    data0 = torch.tensor(data[0], dtype=torch.float32, device=device)
    data1 = torch.tensor(data[1], dtype=torch.float32, device=device)
    A0 = torch.tensor(A[0], dtype=torch.float32, device=device)
    A1 = torch.tensor(A[1], dtype=torch.float32, device=device)

    return data0, data1, A0, A1, aligned_indices, unaligned_indices, labels_0, labels_1








