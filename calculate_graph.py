import numpy as np
import torch
from utils import to_tensor
from sklearn.preprocessing import normalize


def calculate_degree_matrix(similarity_matrix):
    degree_matrix = torch.diag(torch.sum(similarity_matrix, dim=1))

    return degree_matrix


def knn(matrix, k=10, largest=True):
    _, indices = torch.topk(matrix, k=k, dim=1, largest=largest, sorted=True)

    mask = torch.zeros_like(matrix)
    mask.scatter_(1, indices, 1)
    matrix_knn = matrix * mask

    return matrix_knn


def calculate_laplacian(similarity_matrix, k=10):
    similarity_matrix = (similarity_matrix + similarity_matrix.t()) * 0.5
    if k > 0:
        similarity_matrix = knn(similarity_matrix, k=k)

    similarity_matrix = (similarity_matrix + similarity_matrix.t()) * 0.5
    degree_matrix = calculate_degree_matrix(similarity_matrix)
    laplacian_matrix = degree_matrix - similarity_matrix

    return laplacian_matrix


def calculate_cosine_similarity(x1, x2):
    x1 = normalize(x1, axis=1)
    x2 = normalize(x2, axis=1)
    similarity = np.matmul(x1, x2.T)

    return similarity


def calculate_graphs(fea1, fea2, device='cuda'):
    coef1 = calculate_cosine_similarity(fea1, fea1)
    coef2 = calculate_cosine_similarity(fea2, fea2)

    return to_tensor(coef1, device), to_tensor(coef2, device)