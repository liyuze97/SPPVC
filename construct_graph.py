import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def knn_graph_from_similarity(sim_matrix, k):
    N = sim_matrix.shape[0]
    graph = np.zeros((N, N))
    for i in range(N):
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        knn_idx = np.argsort(row)[-k:]
        graph[i, knn_idx] = 1
    graph = np.maximum(graph, graph.T)
    return graph


def normalize_adj_numpy(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def get_graph(data, k, method):
    if method == 'cosine':
        sim = cosine_similarity(data)
    elif method == 'euclidean':
        dist = euclidean_distances(data)
        sim = -dist
    elif method == 'pearson':
        sim = np.corrcoef(data)
        np.fill_diagonal(sim, 0)
        sim = np.clip(sim, 0, 1)
    else:
        raise ValueError("method must be 'cosine' or 'euclidean'")
    graph = knn_graph_from_similarity(sim, k)
    return normalize_adj_numpy(graph)
