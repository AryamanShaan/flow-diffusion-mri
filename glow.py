import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from layers import *
from utils import * 



# --- small helper: channel-reversal as a bijector with zero logdet ---
class ChannelReverse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.perm = torch.arange(channels - 1, -1, -1)

    def forward(self, x):
        # x -> z (inference path)
        return x[:, self.perm.to(x.device)], torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    def inverse(self, z):
        # z -> x (generative path)
        return z[:, self.perm.to(z.device)], torch.zeros(z.size(0), device=z.device, dtype=z.dtype)


# --- one flow step: permutation then coupling (Glow order in your TF code) ---
class FlowStep(nn.Module):
    def __init__(self, x_shape, flow_permutation=1, decomp='NONE', width=128, use_bias_1x1=False):
        """
        x_shape: (C, H, W) at this level (after squeeze)
        """
        super().__init__()
        C, H, W = x_shape

        # permutation
        if flow_permutation == 1:
            self.permute = Conv2d1x1(x_shape=x_shape, bias=use_bias_1x1, decomp=decomp)
            self.perm_type = 'conv1x1'
        elif flow_permutation == 0:
            self.permute = ChannelReverse(C)
            self.perm_type = 'reverse'
        else:
            # No permutation
            self.permute = None
            self.perm_type = 'none'

        # coupling subnet: map pass-through half (C/2,H,W) -> (shift, log_scale) for the other half
        st_net = subnetMLP(x_shape=(C // 2, H, W), hidden_layers=[width, width], shift_only=False)
        self.coupling = AffineCoupling(x_shape=x_shape, shift_and_log_scale_fn=st_net)

        self.C = C
        self.spatial_mult = H * W  # for 1x1 conv ldj multiplier

    def forward_x_to_z(self, x):
        """
        Inference direction: x -> z    (accumulate +ldj)
        Returns: z, ldj (per-batch)
        """
        ldj = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        # permutation
        if self.perm_type == 'conv1x1':
            # 1x1 conv forward (x->z) + ldj = + H*W*log|detA|
            y = self.permute.forward(x)
            ldj = ldj + self.permute._forward_log_det_jacobian(x)
            x = y
        elif self.perm_type == 'reverse':
            x, _ = self.permute.forward(x)
        # else: none

        # coupling forward (x->z): returns transformed tensor; ldj = +sum(s)
        y = self.coupling.forward(x)
        ldj = ldj + self.coupling.forward_log_det_jacobian(x)
        return y, ldj

    def forward_z_to_x(self, z):
        """
        Generative direction: z -> x   (accumulate ldj if you ever need it; here we don't)
        Returns: x
        """
        # coupling inverse (z->x)
        x = self.coupling.inverse(z)
        # permutation inverse (z->x)
        if self.perm_type == 'conv1x1':
            x = self.permute.inverse(x)
        elif self.perm_type == 'reverse':
            x, _ = self.permute.inverse(x)
        return x


class Glow(nn.Module):
    """
    Minimal Glow-like model that mirrors the unconditional path of your TF NoiseFlow:
      - L levels, each: squeeze -> (depth times: permute -> affine coupling)
      - Split after each level except the top; learned conditional split prior via zero-init conv head
      - Top standard normal prior
    """
    def __init__(
        self,
        x_shape,                # (C, H, W) original input shape (NCHW)
        n_levels=2,
        depth=6,
        width=128,
        flow_permutation=1,     # 1: 1x1 conv, 0: reverse, else: none
        decomp='NONE',
        squeeze_factor=2,
        squeeze_type='chessboard',
        use_bias_1x1=False
    ):
        super().__init__()
        self.orig_shape = x_shape
        self.n_levels = n_levels
        self.depth = depth
        self.squeeze_factor = squeeze_factor
        self.squeeze_type = squeeze_type

        # Build levels (track per-level shapes exactly)
        C0, H, W = x_shape
        Cin = C0  # channels BEFORE squeeze at the current level

        self.levels = nn.ModuleList()
        self.split_heads = nn.ModuleList()  # Split2D per level except top

        for level in range(n_levels):
            # After squeeze, flow steps see 4 * Cin channels
            H = H // squeeze_factor
            W = W // squeeze_factor
            C_flow = Cin * 4
            level_shape = (C_flow, H, W)

            steps = nn.ModuleList([
                FlowStep(
                    x_shape=level_shape,
                    flow_permutation=flow_permutation,
                    decomp=decomp,
                    width=width,
                    use_bias_1x1=use_bias_1x1
                )
                for _ in range(depth)
            ])
            self.levels.append(steps)

            if level < n_levels - 1:
                # register split head for this level; next level's Cin = (C_flow)/2
                self.split_heads.append(Split2D(channels=C_flow))
                Cin = C_flow // 2
            else:
                self.split_heads.append(None)
                # top level ends here

    # -------- Loss path: x -> z (inverse in TF naming) --------
    def encode(self, x):
        """
        x: (N, C, H, W)
        Returns: z, objective   where objective = sum of all ldj + top prior logp(z)
        """
        z = x
        objective = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        for level, steps in enumerate(self.levels):
            # squeeze
            z = squeeze2d(z, factor=self.squeeze_factor, squeeze_type=self.squeeze_type)

            # depth flow steps
            for step in steps:
                z, ldj = step.forward_x_to_z(z)
                objective = objective + ldj

            # split (except at top)
            if level < self.n_levels - 1:
                split = self.split_heads[level]
                z, objective = split(z, objective)

        # top prior: standard normal logp
        objective = objective + self._standard_normal_logp(z)
        return z, objective

    # -------- Sampling path: z -> x (forward in TF naming) --------
    def decode(self, z=None, eps_std=None, batch_size=None):
        """
        If z is None: sample from top standard normal with same shape as top latent of a dummy input
        eps_std: optional per-sample std scaling (B,) used only in split reverses (matches TF)
        """
        if z is None:
            C0, H, W = self.orig_shape
            Cin = C0
            for level in range(self.n_levels):
                H = H // self.squeeze_factor
                W = W // self.squeeze_factor
                C_flow = Cin * 4
                if level < self.n_levels - 1:
                    Cin = C_flow // 2  # for next level
                else:
                    top_C = C_flow     # top latent channels
            assert batch_size is not None, "Provide batch_size when sampling with z=None."
            z = torch.randn(batch_size, top_C, H, W, device=next(self.parameters()).device)


        x = z
        for level in reversed(range(self.n_levels)):
            # split reverse (except top)
            if level < self.n_levels - 1:
                split = self.split_heads[level]
                x = split.reverse(x, eps_std=eps_std)

            # steps in reverse order (generative direction)
            steps = self.levels[level]
            for step in reversed(steps):
                x = step.forward_z_to_x(x)

            # unsqueeze
            x = unsqueeze2d(x, factor=self.squeeze_factor, squeeze_type=self.squeeze_type)

        return x

    # -------- Public interfaces mirroring TF names --------
    def inverse(self, x, objective):
        # x -> z, add ldj to objective (no top prior here; TF adds later)
        z = x
        for level, steps in enumerate(self.levels):
            z = squeeze2d(z, factor=self.squeeze_factor, squeeze_type=self.squeeze_type)
            for step in steps:
                z, ldj = step.forward_x_to_z(z)
                objective = objective + ldj
            if level < self.n_levels - 1:
                split = self.split_heads[level]
                z, objective = split(z, objective)
        return z, objective

    def forward(self, z, eps_std=None):
        # z -> x (sampling direction)
        x = z
        for level in reversed(range(self.n_levels)):
            if level < self.n_levels - 1:
                split = self.split_heads[level]
                x = split.reverse(x, eps_std=eps_std)
            steps = self.levels[level]
            for step in reversed(steps):
                x = step.forward_z_to_x(x)
            x = unsqueeze2d(x, factor=self.squeeze_factor, squeeze_type=self.squeeze_type)
        return x

    # -------- Loss wrapper (matches TF .loss) --------
    def loss(self, x):
        """
        Returns mean NLL and sd_z (mean per-sample std of top latent across spatial+channels)
        """
        # objective as (N,)
        objective = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        z, objective = self.inverse(x, objective)  # x->z (ldj + split prior logp)
        # add top prior logp (same as TF ._loss)
        objective = objective + self._standard_normal_logp(z)

        nll = (-objective).mean()

        # sd_z like TF: mean over batch of sqrt(var over C,H,W)
        var_z = z.var(dim=(1, 2, 3), unbiased=False)
        sd_z = (var_z.sqrt()).mean()
        return nll, sd_z

    # -------- Utilities --------
    @staticmethod
    def _standard_normal_logp(z):
        # sum over (C,H,W) → (N,)
        return (-0.5 * (math.log(2 * math.pi) + z ** 2)).flatten(1).sum(dim=1)


# For testing
def main():
    pass

if __name__ == "__main__":
    main()