import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from layers import Conv2dZeros
import math
# from lin_algebra import *
# import scipy as sp

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


# This is not used ?
class GaussianDiag:
    def __init__(self, mean, logsd):
        """
        mean, logsd: tensors of same shape
        """
        self.mean = mean
        self.logsd = logsd
        self.eps = torch.randn_like(mean)  # ε ~ N(0,1)

    def sample(self):
        # z = μ + σ * ε
        return self.mean + torch.exp(self.logsd) * self.eps

    def sample2(self, eps):
        # z = μ + σ * ε (custom ε)
        return self.mean + torch.exp(self.logsd) * eps

    def logps(self, x):
        # per-element log pdf
        return -0.5 * (
            math.log(2 * math.pi)
            + 2.0 * self.logsd
            + (x - self.mean) ** 2 / torch.exp(2.0 * self.logsd)
        )

    def logp(self, x):
        # joint log pdf: sum over all dims except batch
        return self.logps(x).flatten(1).sum(dim=1)

    def get_eps(self, x):
        # recover ε = (x - μ)/σ
        return (x - self.mean) / torch.exp(self.logsd)



class Split2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.prior_head = Conv2dZeros(channels // 2, channels, kernel_size=3, stride=1)

    @staticmethod
    def _logp_diag_gaussian(x, mean, logsd):
        # returns per-sample log-likelihood (N,)
        return (-0.5 * (
            math.log(2 * math.pi)
            + 2.0 * logsd
            + (x - mean) ** 2 / torch.exp(2.0 * logsd)
        )).flatten(1).sum(dim=1)

    def forward(self, z, objective):
        '''
        implements split2d in the original noise flow code
        '''

        N, C, H, W = z.shape
        assert C % 2 == 0, "C must be even"
        C1 = C // 2

        z1, z2 = z[:, :C1], z[:, C1:]          # split along channels
        h = self.prior_head(z1)                # (N, 2*C1, H, W)
        mean = h[:, 0::2]                      # (N, C1, H, W)
        logsd = h[:, 1::2]                     # (N, C1, H, W)

        logsd = torch.clamp(logsd, -5.0, 5.0) # TODO check

        logp = self._logp_diag_gaussian(z2, mean, logsd)  # (N,)
        # Broadcast/add to objective
        if objective.ndim == 0:
            objective += logp.sum()
        else:
            objective += logp

        return z1, objective


    def reverse(self, z1, eps_std=None):
        '''
        implements split2d_reverse in the org noise flow code
        '''
        
        h = self.prior_head(z1)               # (N, 2*C1, H, W)
        mean = h[:, 0::2]
        logsd = h[:, 1::2]

        logsd = torch.clamp(logsd, -5.0, 5.0) #TODO check

        eps = torch.randn_like(mean)
        if eps_std is not None:
            eps = eps * eps_std.view(-1, 1, 1, 1)
        z2 = mean + torch.exp(logsd) * eps
        return torch.cat([z1, z2], dim=1)



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
    '''
    checking split2d
    '''
    # N, C1, H, W = 4, 8, 16, 16
    # z1 = torch.randn(N, C1, H, W)
    # split = Split2D(channels=2*C1)  # module expects full C; it builds head for C/2 = C1

    # # Reverse without eps_std
    # z = split.reverse(z1)                  # -> (N, 2*C1, H, W)
    # print(z.shape)  # torch.Size([4, 16, 16, 16])

    # # Reverse with per-sample noise scaling
    # eps_std = torch.tensor([0.5, 1.0, 2.0, 0.1], dtype=z1.dtype)
    # z_scaled = split.reverse(z1, eps_std=eps_std)  # same shape, different noise scale per sample
    # print(z_scaled.shape)  # torch.Size([4, 16, 16, 16])

    # ******************************************************************
    pass

if __name__ == "__main__":
    main()