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
        
        return x

    def _forward_log_det_jacobian(self, x=None):
        return -self.inverse_log_det_jacobian()

    def _inverse_log_det_jacobian(self, y=None):
        _, _, log_abs_det = self._assemble_A()   # log|det A|
        return log_abs_det * (self.i0 * self.i1)

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

        self.C_in, self.H, self.W = x_shape # not doing C_total/2 since input has just one channel and mask used in AffineCoupling

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

    def __init__(self, x_shape, shift_and_log_scale_fn, layer_id=0, last_layer=False,validate_args=False):
        super().__init__()
        pass



    

# For testing
def main():
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

    # -------------------------------------------------------
    # Create the subnet for MNIST-like input (1 channel, 28x28)
    subnet = subnetMLP(x_shape=[1, 28, 28],
                            hidden_layers=[64, 64],
                            shift_only=False)

    # Fake MNIST batch: 8 samples of 1×28×28
    x = torch.randn(8, 1, 28, 28, requires_grad=True)

    # Forward pass
    shift, log_scale = subnet(x)

    print("Input shape:    ", x.shape)
    print("Shift shape:    ", shift.shape)
    print("Log_scale shape:", log_scale.shape)

    # Simple scalar loss to check gradients flow
    loss = (shift.mean() + log_scale.mean())
    loss.backward()

    print("\nGradient on input? ", x.grad is not None)
    print("Gradient on first linear weight? ", subnet.mlp[0].weight.grad is not None)

if __name__ == "__main__":
    main()