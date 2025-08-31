import torch
from torch import nn
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
        np_log_s = np.log(abs(np_s))
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

        if self.bias:
            pass

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


    def forward(self, x):
        pass

    def inverse(self, y):
        pass

    def _forward_log_det_jacobian(self, x):
        pass

    def _inverse_log_det_jacobian(self, y):
        pass

    def _forward_and_log_det_jacobian(self, x):
        pass

    def _inverse_and_log_det_jacobian(self, y):
        pass

    def _inverse_and_log_det_jacobian___(self, z, y):
        pass
        

    

# For testing
def main():
    pass

if __name__ == "__main__":
    main()