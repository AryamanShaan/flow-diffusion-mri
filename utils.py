import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from lin_algebra import *
import scipy as sp

def squeeze2d(x, factor=2, squeeze_type='chessboard'):
    """
    x: (N, C, H, W)
    Returns: (N, C * factor^2, H/factor, W/factor)
    """
    assert factor >= 1
    if factor == 1:
        return x

    N, C, H, W = x.shape
    assert H % factor == 0 and W % factor == 0

    if squeeze_type == 'chessboard':
        # Reshape: split H and W into (H/f, f), (W/f, f)
        x = x.reshape(N, C, H // factor, factor, W // factor, factor)
        # Permute to bring factors next to channel dim
        x = x.permute(0, 1, 3, 5, 2, 4)  # (N, C, f, f, H/f, W/f)
    elif squeeze_type == 'patch':
        # Reshape: factor-first
        x = x.reshape(N, C, factor, H // factor, factor, W // factor)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (N, C, f, f, H/f, W/f)
    else:
        raise ValueError(f"Unknown squeeze type: {squeeze_type}")

    # Merge factor dims into channels
    x = x.reshape(N, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(x, factor=2, squeeze_type='chessboard'):
    """
    x: (N, C * factor^2, H, W)
    Returns: (N, C, H*factor, W*factor)
    """
    assert factor >= 1
    if factor == 1:
        return x

    N, C, H, W = x.shape
    assert C % (factor * factor) == 0
    C //= factor * factor

    # Split channels back into (C, f, f)
    x = x.reshape(N, C, factor, factor, H, W)

    if squeeze_type == 'chessboard':
        x = x.permute(0, 1, 4, 2, 5, 3)  # (N, C, H, f, W, f)
    elif squeeze_type == 'patch':
        x = x.permute(0, 1, 2, 4, 3, 5)  # (N, C, f, H, f, W)
    else:
        raise ValueError(f"Unknown squeeze type: {squeeze_type}")

    # Merge back factors with spatial dims
    x = x.reshape(N, C, H * factor, W * factor)
    return x





# For testing
def main():

    # ******************************************************************
    '''
    checking squeeze2d and unsqueeze 2d
    '''
    # x = torch.randn(2, 3, 8, 8)  # random values
    # y = squeeze2d(x, factor=2, squeeze_type="chessboard")
    # z = unsqueeze2d(y, factor=2, squeeze_type="chessboard")

    # # 1. Shape equality
    # print("Original shape:", x.shape)
    # print("Squeezed shape:", y.shape)
    # print("Restored shape:", z.shape)

    # # 2. Exact value preservation
    # if torch.equal(x, z):
    #     print("✅ Values are exactly preserved (bitwise equal)")
    # else:
    #     # if using float ops with GPU, sometimes tiny rounding occurs → use allclose
    #     max_diff = (x - z).abs().max().item()
    #     print(f"⚠️ Values differ slightly, max diff = {max_diff}")
    #     assert torch.allclose(x, z, atol=1e-8), "Values changed!"

    # ******************************************************************
    pass

if __name__ == "__main__":
    main()