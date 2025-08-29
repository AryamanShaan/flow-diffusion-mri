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


def vec2stricttri(matrix, upper:bool):
    # assume matrix is a tensor
    matrix = torch.as_tensor(matrix) # TODO check if this line is alright
    if matrix.shape[-1:] == [0]:
        return matrix
    
    batch_shape = matrix.shape[:-1]

    matrix = matrix.reshape(-1, matrix.size(-1)) # converting torch.Size([2, 3, 4, 5]) to torch.Size([24, 5])

    M_tri_base = fill_triangular(matrix, upper=upper)

    if upper:
        result = torch.nn.functional.pad(M_tri_base, (1, 0, 0, 1))  # (left, right, top, bottom)
    else:
        result = torch.nn.functional.pad(M_tri_base, (0, 1, 1, 0))
    
    final_shape = batch_shape + result.shape[-2:]
    return result.reshape(final_shape)


def stricttri2vec(matrix, upper:bool):
    matrix = torch.as_tensor(matrix) # TODO check if this line is alright
    if tuple(matrix.shape[-2:]) == (0, 0):
        return matrix

    batch_shape = matrix.shape[:-2]

    r, c = matrix.shape[-2], matrix.shape[-1]
    matrix = matrix.reshape(-1, r, c)

    if upper:
        # remove last row, first column
        matrix_trim = matrix[..., :-1, 1:]
        # keep upper-triangular (incl. main diagonal)
        matrix_tri = torch.triu(matrix_trim, diagonal=0)
    else:
        # remove first row, last column
        matrix_trim = matrix[..., 1:, :-1]
        # keep lower-triangular (incl. main diagonal)
        matrix_tri = torch.tril(matrix_trim, diagonal=0)

    # early exit if empty 0x0
    if tuple(matrix_trim.shape[-2:]) == (0, 0):
        return matrix_trim

    result = fill_triangular_inverse(matrix_tri, upper=upper)    

    final_shape = batch_shape + (result.shape[-1],)
    ret = result.reshape(final_shape)   
    return ret


def fill_triangular(x, upper=False):
    '''
    torch implementation of tfp.math.fill_triangular
    technically noise-flow implemented tensorflow.contrib.distributions.fill_triangular
    but tensorflow.contrib.distributions got deprecated and replaced by tfp (tensorflow-probability)
    '''
    x = torch.as_tensor(x)
    if x.ndim < 1:
        raise ValueError("x must have rank >= 1")
    m = x.shape[-1]
    if m is not None: # this 'if' condition will always execute. original tf code allowed for dynamic 
        # Formula derived by solving for n: m = n(n+1)/2.
        m = np.int32(m)
        n = np.sqrt(0.25 + 2. * m) - 0.5
        if n != np.floor(n):
            raise ValueError('Input right-most shape ({}) does not '
                                'correspond to a triangular matrix.'.format(m))
        n = np.int32(n)
        batch_shape = x.shape[:-1]
        final_shape = batch_shape + (n, n)
    else: # this else will actually never execute, torch does not have dynamic
        m = torch.tensor(x.shape[-1], dtype=torch.int32)  # dynamic last dim
        n = torch.sqrt(0.25 + (2*m).to(torch.float32))
        n = n.to(torch.int32)   # cast back to int32
        batch_shape = x.shape[:-1]
        final_shape = batch_shape + (n, n) 
    
    ndims = x.ndim
    if upper:
        x_list = [x, torch.flip(x[..., n:], dims=[-1])]
    else:
        x_list = [x[..., n:], torch.flip(x, dims=[-1])]
    new_shape = x.shape[:-1] + (n, n)
    x = torch.cat(x_list, dim=-1).reshape(new_shape)
    if upper:
        x = torch.triu(x, diagonal=0)  # keep upper triangle incl diag
    else:
        x = torch.tril(x, diagonal=0)  # keep lower triangle incl diag
    
    return x


def fill_triangular_inverse(x, upper=False):
    '''
    torch implementation of tfp.math.fill_triangular_inverse
    technically noise-flow implemented tensorflow.contrib.distributions.fill_triangular_inverse
    but tensorflow.contrib.distributions got deprecated and replaced by tfp (tensorflow-probability)
    '''
    x = torch.as_tensor(x)
    if x.ndim < 2:
        raise ValueError('x must have rank 2')
    n = x.shape[-1]
    m = (n*(n+1)) // 2
    final_shape = x.shape[:-2] + (m,)
    ndims = x.ndim
    if upper:
        initial_elements = x[..., 0, :]                         # first row
        triangular_portion = x[..., 1:, :]                      # remaining rows
    else:
        initial_elements = torch.flip(x[..., -1, :], dims=[ndims - 2]) # last row, reversed
        triangular_portion = x[..., :-1, :]                      # all but last row
    # Double reverse operation
    rotated_triangular_portion = torch.flip(
        torch.flip(triangular_portion, dims=[ndims - 1]), dims=[ndims - 2])
    consolidated_matrix = triangular_portion + rotated_triangular_portion
    # Reshape operation
    batch_shape = x.shape[:-2]
    new_shape = batch_shape + (n * (n - 1),)
    end_sequence = consolidated_matrix.reshape(new_shape)
    # Concatenation
    y = torch.cat([initial_elements, end_sequence[..., :m - n]], dim=-1)
    # Note: PyTorch doesn't have static shape setting like TensorFlow
    # The shape is automatically tracked at runtime
    return y


# For testing
def main():
    x = vec2stricttri([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15], False)
    print(x)
    print()
    x= stricttri2vec(x, False)
    print(x)

if __name__ == "__main__":
    main()