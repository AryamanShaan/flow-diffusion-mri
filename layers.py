import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from lin_algebra import *
import scipy as sp


class Conv2d1x1(nn.Module):

    def __init__(self, x_shape, bias=True, decomp='NONE', last_layer=False, 
                 layer_id=0, order=0):
        super().__init__()
        
        # Store configuration
        self.x_shape = x_shape
        self.ic, self.i0, self.i1 = x_shape  # C, H, W order
        self._bias = bias
        self._decomp = decomp
        self.id = layer_id
        self._last_layer = last_layer
        self._order = order
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):     # also implements matrix_params_lu in noise-flow repo
        w_shape = [self.ic, self.ic]
        self.w_shape = w_shape
        np_w = sp.linalg.qr(np.random.randn(*w_shape))[0].astype('float32') # get random orthogonal matrix, always invertible

        ########################
        # matrix_param_lu part starts

        init_A = np_w
        np_p, np_l, np_u = sp.linalg.lu(init_A)

        np_s = np.diag(np_u)
        np_sign_s = np.sign(np_s)
        np_log_s = np.log(np.abs(np_s))
        np_u = np.triu(np_u, k=1)

        # P is constant (not trainable)
        self.register_buffer("p", torch.tensor(np_p, dtype=torch.float32))
        # sign_s is constant (±1 values, not trainable)
        self.register_buffer("sign_s", torch.tensor(np_sign_s, dtype=torch.float32))
        # log_s is trainable (learned parameter)
        self.log_s = nn.Parameter(torch.tensor(np_log_s, dtype=torch.float32))

        init_l_vec = stricttri2vec(torch.tensor(np_l, dtype=torch.float32), upper=False)
        self.l_vec = nn.Parameter(init_l_vec.clone())

        init_u_vec = stricttri2vec(torch.tensor(np_u, dtype=torch.float32), upper=True)
        self.u_vec = nn.Parameter(init_u_vec.clone()) 
        
        # matrix_params_lu part ends
        ########################

        self.bias = None
        if self._bias:
            self.bias = nn.Parameter(torch.zeros(self.ic))

    def _assemble_A(self):
        '''
        part of matrix_params_lu
        '''
        l_base = vec2stricttri(self.l_vec, upper=False)   # (C, C), diag=0, below-diag = params
        I = torch.eye(self.ic, device=l_base.device, dtype=l_base.dtype) # add ones to the diagonal
        L = l_base + I # add ones to the diagonal

        u_base = vec2stricttri(self.u_vec, upper=True)     
        diag = self.sign_s * torch.exp(self.log_s)       
        U = u_base + torch.diag(diag)
        
        A = self.p @ L @ U

        # P_inv = P^T
        p_inv = self.p.t()
        # Solve L X = P_inv  (L lower-triangular)
        X = torch.linalg.solve_triangular(L, p_inv, upper=False)
        # Solve U A_inv = X  (U upper-triangular)
        A_inv = torch.linalg.solve_triangular(U, X, upper=True)

        log_abs_det = self.log_s.sum()

        return A, A_inv, log_abs_det


    def inverse(self, x):
        '''
        This is actually the 'forward' function in flow model terminology || z -> x || applies A_inv || used for sampling
        '''
        if self._last_layer:
            n = x.shape[0]
            x = x.view(n, self.ic, self.i0, self.i1)  # NCHW
        if self._bias and self.bias is not None:
            x = x - self.bias.view(1, -1, 1, 1)

        _, A_inv, _ = self._assemble_A() # Assemble A and its inverse (A_inv is CxC)
        W = A_inv.view(self.ic, self.ic, 1, 1)  # PyTorch conv2d wants (outC, inC, kH, kW) = (C, C, 1, 1)
        y = F.conv2d(x, W, bias=None) # 1x1 convolution applying the channel-mixing A^{-1}

        return y
    

    def forward(self, y):
        '''
        This is actually the 'inverse' function in flow model terminology || x -> z || applies A || used for loss
        '''
        A, _, ladj = self._assemble_A()   # A is (C, C)
        W = A.view(self.ic, self.ic, 1, 1)     # conv weight (outC, inC, 1, 1)
        x = F.conv2d(y, W, bias=None)           # 1x1 channel mixing

        if self._bias and self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)            # add per-channel bias

        if self._last_layer:
            x = x.reshape(x.size(0), self.ic * self.i0 * self.i1)  # flatten
        
        return x, ladj * (self.i0 * self.i1)

    def _forward_log_det_jacobian(self, x=None): # not used 
        _, _, log_abs_det = self._assemble_A()   # log|det A|
        return log_abs_det * (self.i0 * self.i1)


    def _inverse_log_det_jacobian(self, y=None): # not used
        _, _, log_abs_det = self._assemble_A()   # log|det A|
        return -log_abs_det * (self.i0 * self.i1)

    def _forward_and_log_det_jacobian(self, x): 
        pass

    def _inverse_and_log_det_jacobian(self, y):
        pass

    def _inverse_and_log_det_jacobian___(self, z, y):
        pass


class subnetMLP(nn.Module):
    """
    PyTorch port of the TF 'real_nvp_default_template' (channel-split version) in noise-flow repo.
    Input:  x of shape (N, C_in, H, W) where C_in = C_total // 2
    Output: (shift, log_scale) each (N, C_in, H, W)  OR (shift, None) if shift_only=True
    """
    def __init__(self,
                 x_shape,                # [C, H, W] - no batch size
                 hidden_layers,          # e.g., [512, 512]
                 shift_only: bool = False):
        super().__init__()
        self.shift_only = shift_only

        # H, W, C_total = x_shape
        # C_in = C_total // 2
        # self.H, self.W, self.C_in = H, W, C_in

        self.C_in, self.H, self.W = x_shape # not doing C_total/2 since input may have one channel 

        in_features = self.C_in * self.H * self.W

        layers = []
        prev = in_features
        for i, units in enumerate(hidden_layers):
            layers.append(nn.Linear(prev, units, bias=True))
            layers.append(nn.BatchNorm1d(units))           # BN after dense (mirrors TF BN)
            layers.append(nn.ReLU())                # activation
            prev = units
        self.mlp = nn.Sequential(*layers)

        out_features = in_features if shift_only else 2 * in_features
        self.proj = nn.Linear(prev, out_features, bias=True)

        # ---- initialization ----
        # zero-init final layer (identity start for coupling)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # He init hidden linears (good for ReLU-family)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


    def forward(self, x: torch.Tensor):
        """
        x: (N, C_in, H, W)  
        """
        N, C, H, W = x.shape
        assert (C, H, W) == (self.C_in, self.H, self.W), \
            f"Expected (N,{self.C_in},{self.H},{self.W}), got {tuple(x.shape)}"

        x_flat = x.reshape(N, -1)            # (N, H*W*C_in)
        h = self.mlp(x_flat)            # (N, hidden_last)
        out = self.proj(h)              # (N, (1 or 2)*H*W*C_in)

        if self.shift_only:
            shift = out.view(N, C, H, W)
            log_scale = None
            return shift, log_scale

        out = out.reshape(N, 2 * C, H, W)
        shift, log_scale = torch.split(out, [C, C], dim=1)
        return shift, log_scale
        



class AffineCoupling(nn.Module):

    def __init__(self,
                 x_shape,                     
                 shift_and_log_scale_fn,      # callable: x0 -> (shift, log_scale), both shaped like x1
                 layer_id: int = 0,
                 last_layer: bool = False,
                 name: str = "real_nvp"):
        super().__init__()
        self.x_shape = x_shape
        self.ic, self.i0, self.i1  = x_shape  
        self._last_layer = last_layer
        self.id = layer_id
        self._shift_and_log_scale_fn = shift_and_log_scale_fn # instance of subnetMLP

        # Learnable scalar for tempering log_scale: scale * tanh(log_scale)
        self.scale = nn.Parameter(torch.tensor(1e-4, dtype=torch.float32))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        This is actually the 'inverse' function in flow model terminology || x -> z || used for loss
        '''
        # x is assumed NCHW 

        C = x.shape[1]
        assert C % 2 == 0, f"Coupling split needs even channels, got C={C}"
        x0, x1 =  x[:, :C // 2, ...], x[:, C // 2:, ...]
        # assert C == self.ic, f"Expected channel={self.ic}, got {C}"

        shift, log_scale = self._shift_and_log_scale_fn(x0)
        assert shift.shape == x1.shape and (log_scale is None or log_scale.shape == x1.shape)
        if log_scale is not None:
            log_scale = self.scale * torch.tanh(log_scale)
            log_scale = torch.clamp(log_scale, -5.0, 5.0) #TODO check

        y1 = x1
        if log_scale is not None:
            y1 = y1 * torch.exp(log_scale)
        if shift is not None:
            y1 = y1 + shift

        x = torch.cat([x0, y1], dim=1)
        if self._last_layer and x.dim() == 4:
            N = x.shape[0]
            x = x.contiguous().view(N, -1)  

        if log_scale is None:
            log_scale =  torch.zeros(x.size(0), dtype=x.dtype, device=x.device)


        return x, log_scale.flatten(1).sum(dim=1)


    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        '''
        This is actually the 'forward' function in flow model terminology || z - > x || used for sampling
        '''
        if self._last_layer and y.dim() == 2: # if last_layer and 2D, reshape to NCHW
            N = y.shape[0]
            C, H, W = self.ic, self.i0, self.i1
            y = y.view(N, C, H, W).contiguous()  
        C = y.shape[1]
        assert C % 2 == 0, f"Coupling split needs even channels, got C={C}"
        y0, y1 =  y[:, :C // 2, ...], y[:, C // 2:, ...]
        # assert C == self.ic, f"Expected channel={self.ic}, got {C}"

        shift, log_scale = self._shift_and_log_scale_fn(y0)
        assert shift.shape == y1.shape and (log_scale is None or log_scale.shape == y1.shape)
        if log_scale is not None:
            log_scale = self.scale * torch.tanh(log_scale)
            log_scale = torch.clamp(log_scale, -5.0, 5.0) #TODO check

        x1 = y1
        if shift is not None:
            x1 = x1 - shift
        if log_scale is not None:
            x1 = x1 * torch.exp(-log_scale)

        x = torch.cat([y0, x1], dim=1)
        return x


    def forward_log_det_jacobian(self, z: torch.Tensor) -> torch.Tensor:

        C = z.shape[1]
        assert C % 2 == 0, f"Coupling split needs even channels, got C={C}"
        z0, _ =  z[:, :C // 2, ...], z[:, C // 2:, ...]
        # assert C == self.ic, f"Expected channel={self.ic}, got {C}"

        _, log_scale = self._shift_and_log_scale_fn(z0)
        assert log_scale is None or log_scale.shape == z0.shape
        if log_scale is not None:
            log_scale = self.scale * torch.tanh(log_scale)

        if log_scale is None:
            return torch.zeros(z.size(0), dtype=z.dtype, device=z.device)

        return log_scale.flatten(1).sum(dim=1)
    

    def inverse_log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor: # not used 

        # if last_layer and 2D, reshape to NCHW
        C = x.shape[1]
        assert C % 2 == 0, f"Coupling split needs even channels, got C={C}"
        x0, _ =  x[:, :C // 2, ...], x[:, C // 2:, ...]
        # assert C == self.ic, f"Expected channel={self.ic}, got {C}"

        _, log_scale = self._shift_and_log_scale_fn(x0)
        assert log_scale is None or log_scale.shape == x0.shape
        if log_scale is not None:
            log_scale = self.scale * torch.tanh(log_scale)

        if log_scale is None:
            return torch.zeros(x.size(0), dtype=x.dtype, device=x.device)

        # sum over transformed half: (N, C/2, H, W) -> (N,)
        return -log_scale.flatten(1).sum(dim=1)


    def forward_and_log_det_jacobian(self, x: torch.Tensor): # not used
        pass

    def inverse_and_log_det_jacobian(self, y: torch.Tensor):
        pass


class Conv2dZeros(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, logscale_factor=1.0): # TODO check logscale_factor 1.0 or 3.0?
        super().__init__()
        # Conv layer with zero-initialized weights & bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        # Per-channel log-scale parameter
        self.logscale = nn.Parameter(torch.zeros(out_channels))
        self.logscale_factor = logscale_factor

    def forward(self, x):
        out = self.conv(x)
        scale = torch.exp(self.logscale * self.logscale_factor).view(1, -1, 1, 1)
        return out * scale


class ActNorm2d(nn.Module): # TODO check this code with some official code

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.initialized = False
        self.bias  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.eps = eps

    @torch.no_grad()
    def _init_from_data(self, x):
        # x: (N,C,H,W)
        mean = x.mean(dim=[0,2,3], keepdim=True)
        var  = x.var (dim=[0,2,3], unbiased=False, keepdim=True)
        std  = (var + self.eps).sqrt()
        self.bias.data  = -mean
        self.scale.data = 1.0 / std
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init_from_data(x)

        # ldj per sample (N,)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        # scale is (1,C,1,1), log|scale| sum over C then * H*W
        ldj_per_channel = torch.log(torch.abs(self.scale)).view(-1)  # C elements
        ldj_scalar = (H * W) * ldj_per_channel.sum()

        return self.scale * (x + self.bias), ldj_scalar.expand(x.size(0)) 
    
    def inverse(self, y):
        if not self.initialized:
            # If called before forward, we can’t init from data; assume identity
            return y
        return y / self.scale - self.bias

    def forward_log_det_jacobian(self, x):
        # ldj per sample (N,)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        # scale is (1,C,1,1), log|scale| sum over C then * H*W
        ldj_per_channel = torch.log(torch.abs(self.scale)).view(-1)  # C elements
        ldj_scalar = (H * W) * ldj_per_channel.sum()
        return ldj_scalar.expand(x.size(0))  # same for all samples in batch

    

    

class SignalDependentLayer(nn.Module): # this is not correct maybe, might need to fix

    def __init__(self, x_shape, eps: float = 1e-8):
        super().__init__()
        self.C, self.H, self.W = x_shape
        self.eps = eps

        # Learnable scalars b1, b2 -> beta1, beta2. Betas need to be positive
        self.b1 = nn.Parameter(torch.tensor(-5.0, dtype=torch.float32))  
        self.b2 = nn.Parameter(torch.tensor( 0.0, dtype=torch.float32))  
    

    def forward(self, x: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        '''
        x -> z || used for training
        '''
        beta1 = torch.exp(self.b1)
        beta2 = torch.exp(self.b2)
        # numerical safety: inside sqrt and log should be > 0
        inside = beta1 * I + beta2
        # clamp to avoid negative due to numerical noise when beta1 ~ 0
        inside = torch.clamp_min(inside, self.eps)
        s = torch.sqrt(inside)
        log_s = torch.log(torch.clamp_min(s, self.eps))
        log_det_jacobian = log_s.flatten(1).sum(dim=1) 
        return s * x, log_det_jacobian

    def inverse(self, y: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        '''
        z -> x || used for sampling
        '''
        beta1 = torch.exp(self.b1)
        beta2 = torch.exp(self.b2)
        # numerical safety: inside sqrt and log should be > 0
        inside = beta1 * I + beta2
        # clamp to avoid negative due to numerical noise when beta1 ~ 0
        inside = torch.clamp_min(inside, self.eps)
        s = torch.sqrt(inside)
        log_s = torch.log(torch.clamp_min(s, self.eps))
        log_det_jacobian = log_s.flatten(1).sum(dim=1) 
        return y / s, -log_det_jacobian





# For testing
def main():

    # ******************************************************************
    '''
    Testing conv2d1x1
    '''
    # C,H,W = 4, 5, 6
    # layer = Conv2d1x1((C,H,W), bias=True, last_layer=False, decomp='LU')  # you currently only implement LU
    # inp = torch.randn(2, C, H, W)
    # z = layer(inp)                # forward: data -> latent
    # rec = layer.inverse(z)        # latent -> data
    # print("recon err:", (inp - rec).abs().max().item())
    # print("logdet inverse:", layer._inverse_log_det_jacobian().item())

    # # A * A_inv ≈ I ?
    # A, A_inv, ladj = layer._assemble_A()
    # print("||A A_inv - I||:", torch.norm(A @ A_inv - torch.eye(A.size(0))).item())

    # # forward & inverse logdets should be negatives
    # ladj_inv = layer._inverse_log_det_jacobian()
    # ladj_fwd = -ladj_inv
    # print("ladj:", ladj.item(), " ladj_inv:", ladj_inv.item(), " ladj_fwd:", ladj_fwd.item())

    # # gradient sanity: make sure params get grads
    # x = torch.randn(2, layer.ic, layer.i0, layer.i1)
    # z = layer(x)
    # loss = z.pow(2).mean()
    # loss.backward()
    # print("has grad l_vec/u_vec/log_s:",
    #   layer.l_vec.grad is not None,
    #   layer.u_vec.grad is not None,
    #   layer.log_s.grad is not None)

    # ******************************************************************
    '''
    Testing subnet
    '''
    # Create the subnet for MNIST-like input (1 channel, 28x28)
#     subnet = subnetMLP(x_shape=[1, 28, 28],
#                             hidden_layers=[64, 64],
#                             shift_only=False)

#     # Fake MNIST batch: 8 samples of 1×28×28
#     x = torch.randn(8, 1, 28, 28, requires_grad=True)

#     # Forward pass
#     shift, log_scale = subnet(x)

#     print("Input shape:    ", x.shape)
#     print("Shift shape:    ", shift.shape)
#     print("Log_scale shape:", log_scale.shape)

#     # Simple scalar loss to check gradients flow
#     loss = (shift.mean() + log_scale.mean())
#     loss.backward()

#     print("\nGradient on input? ", x.grad is not None)
#     print("Gradient on first linear weight? ", subnet.mlp[0].weight.grad is not None)

    # ******************************************************************
    '''
    Testing AffineCoupling + subnet
    '''
    # torch.manual_seed(0)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Fake MNIST-like: N=8, C_total=2 (even, so channel-split works), H=W=28
    # N, C_total, H, W = 8, 2, 28, 28
    # x = torch.randn(N, C_total, H, W, device=device, requires_grad=True)

    # # subnet must be built for HALF the channels (C_in = C_total//2)
    # st_net = subnetMLP(x_shape=(C_total // 2, H, W), hidden_layers=[512, 512], shift_only=False).to(device)
    # st_net.train()  # enable BN training mode for this test

    # # coupling built with full (C, H, W)
    # coupling = AffineCoupling(
    #     x_shape=(C_total, H, W),
    #     shift_and_log_scale_fn=st_net,
    #     last_layer=False
    # ).to(device)

    # # ---------- forward: x -> z ----------
    # z = coupling.forward(x)
    # assert z.shape == x.shape, f"z shape {z.shape} != x shape {x.shape}"

    # fldj = coupling.forward_log_det_jacobian(x)  # log|det J_f(x)|
    # assert fldj.shape == (N,), f"fldj shape {fldj.shape} != {(N,)}"

    # # ---------- inverse: z -> x_recon ----------
    # x_recon = coupling.inverse(z)
    # recon_err = (x_recon - x).abs().max().item()
    # print(f"recon error || inverse(forward(x)) - x ||_inf = {recon_err:.3e}")
    # # expect ~1e-6 to 1e-7 once BN settles; with BN in train mode, small noise is OK

    # # ---------- inverse logdet vs forward logdet ----------
    # ildj = coupling.inverse_log_det_jacobian(z)  # log|det J_f^{-1}(z)|
    # # For exact arithmetic, ildj should be -fldj (elementwise)
    # sign_err = (ildj + fldj).abs().max().item()
    # print(f"max | ildj + fldj | = {sign_err:.3e}")

    # # ---------- simple backward test ----------
    # # Use a tiny base logprob to mimic NLL: L = -( -0.5 * z^2 ).mean() - fldj.mean()
    # # (Not a real training loss, just to check gradients flow)
    # fake_ll = (-0.5 * z.pow(2)).mean() + fldj.mean()  # log p(z) + log|detJ|
    # loss = -fake_ll
    # loss.backward()

    # # Check a couple of grads
    # has_x_grad = x.grad is not None and torch.isfinite(x.grad).all().item()
    # has_st_grad = st_net.mlp[0].weight.grad is not None and torch.isfinite(st_net.mlp[0].weight.grad).all().item()
    # print(f"grad x        : {'OK' if has_x_grad else 'MISSING'}")
    # print(f"grad st_net w0: {'OK' if has_st_grad else 'MISSING'}")

    # # ---------- also test last_layer flatten/reshape path ----------
    # coupling_last = AffineCoupling(
    #     x_shape=(C_total, H, W),
    #     shift_and_log_scale_fn=st_net,
    #     last_layer=True
    # ).to(device)

    # # forward with last_layer=True should flatten the output
    # z_flat = coupling_last.forward(x.detach().clone().requires_grad_(True))
    # print("z_flat shape (last_layer=True, forward):", tuple(z_flat.shape))
    # assert z_flat.dim() == 2 and z_flat.shape[1] == C_total * H * W

    # # inverse should accept flat and return NCHW
    # x_recon2 = coupling_last.inverse(z_flat)
    # print("x_recon2 shape (last_layer=True, inverse):", tuple(x_recon2.shape))
    # assert x_recon2.shape == (N, C_total, H, W)

    # print("\nAll tests ran.")

    # ******************************************************************
    '''
    Testing Conv2dZeros
    '''
    # x = torch.randn(4, 3, 16, 16)

    # # Define conv2d_zeros equivalent
    # conv = Conv2dZeros(in_channels=3, out_channels=8, kernel_size=3)

    # # Forward pass
    # y = conv(x)

    # # Print shapes
    # print("Input shape :", x.shape)   # (4, 3, 16, 16)
    # print("Output shape:", y.shape)   # (4, 8, 16, 16)

    # # Check if output is zero at initialization
    # print("Max abs value in output:", y.abs().max().item())
    # print("Are all outputs zero?", torch.allclose(y, torch.zeros_like(y)))

    # ******************************************************************
    pass


if __name__ == "__main__":
    main()