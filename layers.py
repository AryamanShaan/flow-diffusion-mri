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
        

    

# For testing
def main():
    C,H,W = 4, 5, 6
    layer = Conv2d1x1((C,H,W), bias=True, last_layer=False, decomp='LU')  # you currently only implement LU
    inp = torch.randn(2, C, H, W)
    z = layer(inp)                # forward: data -> latent
    rec = layer.inverse(z)        # latent -> data
    print("recon err:", (inp - rec).abs().max().item())
    print("logdet inverse:", layer._inverse_log_det_jacobian().item())

    # A * A_inv ≈ I ?
    A, A_inv, ladj = layer._assemble_A()
    print("||A A_inv - I||:", torch.norm(A @ A_inv - torch.eye(A.size(0))).item())

    # forward & inverse logdets should be negatives
    ladj_inv = layer._inverse_log_det_jacobian()
    ladj_fwd = -ladj_inv
    print("ladj:", ladj.item(), " ladj_inv:", ladj_inv.item(), " ladj_fwd:", ladj_fwd.item())

    # gradient sanity: make sure params get grads
    x = torch.randn(2, layer.ic, layer.i0, layer.i1)
    z = layer(x)
    loss = z.pow(2).mean()
    loss.backward()
    print("has grad l_vec/u_vec/log_s:",
      layer.l_vec.grad is not None,
      layer.u_vec.grad is not None,
      layer.log_s.grad is not None)



if __name__ == "__main__":
    main()