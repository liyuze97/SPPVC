import numpy as np
import torch
import numpy.linalg as lg

import warnings
warnings.filterwarnings('ignore')

def sqrtm_torch(A_np, device='cuda'):
    A = torch.from_numpy(A_np).float().to(device)
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=1e-8)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

def wasserstein_initialisation(A, B, device='cuda'):
    C1_tilde = sqrtm_torch(A, device)
    C2_tilde = sqrtm_torch(B, device)
    return [C1_tilde, C2_tilde]

def regularise_and_invert(x, y, alpha, ones):
    x_reg = regularise_invert_one(x, alpha, ones)
    y_reg = regularise_invert_one(y, alpha, ones)
    return [x_reg, y_reg]

def regularise_invert_one(x, alpha, ones):
    if ones:
        x_reg = lg.inv(x.cpu().numpy() + alpha * np.eye(len(x)) + np.ones([len(x), len(x)]) / len(x))
    else:
        x_reg = lg.pinv(x) + alpha * np.eye(len(x))
    return x_reg


