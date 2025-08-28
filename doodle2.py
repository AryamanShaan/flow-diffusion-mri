import torch
from torch import nn
import numpy as np

def flatten_everything_but_c():
    # Example tensor (B, H, W, C) like in TF (NHWC)
    B, H, W, C = 2, 3, 4, 5
    x = torch.arange(B * H * W * C).reshape(B, H, W, C)
    print(x.shape[-1:])
    print("Original shape (NHWC):", x.shape)   # torch.Size([2, 3, 4, 5])

    # Mimic the TF reshape: flatten all but the last dimension
    reshaped = x.view(-1, x.shape[-1])

    print("Reshaped shape:", reshaped.shape)   # torch.Size([24, 5])


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



def main():

    # flatten_everything_but_c()
    # -----------------------------
    # x = torch.tensor([1, 2, 3, 4, 5, 6], dtype=int)
    # L = fill_triangular(x, upper=False)
    # U = fill_triangular(x, upper=True)
    # print("Lower:\n", L)
    # print("Upper:\n", U)
    # -----------------------------
    # m = 8
    # n = np.sqrt(0.25 + 2. * m) - 0.5
    # print(n)
    # -----------------------------
    x = fill_triangular([1, 2, 3, 4, 5, 6], True)
    print(x)
    print(fill_triangular_inverse(x, True))
    print()
    x = fill_triangular([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], True)
    print(x)
    print(fill_triangular_inverse(x, True))
    print()
    x = fill_triangular([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], True)
    print(x)
    print(fill_triangular_inverse(x, True))
    print()

if __name__ == "__main__":
    main()