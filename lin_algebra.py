import numpy as np
import scipy as sp
import torch
from torch import nn


# matrix_param_none function is not really used
def matrix_param_none(init_A, dtype=torch.float32, device=None):
    A = torch.nn.Parameter(torch.as_tensor(init_A, dtype=dtype, device=device))
    A_inv = torch.linalg.inv(A)
    sign, log_abs_det = torch.linalg.slogdet(A)  # TODO firgure out if sign is not used?
    return {'A': A, 'A_inv': A_inv, 'log_abs_det': log_abs_det}


def vec2stricttri(matrix):
    # assume matrix is a tensor
    if matrix.shape[-1:] == [0]:
        return matrix
    matrix = matrix.view(-1, matrix.size(-1)) # converting torch.Size([2, 3, 4, 5]) to torch.Size([24, 5])
    
