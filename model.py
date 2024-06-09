import torch
from torch import nn
from torch.nn import functional as F

class ESN(nn.Module):
    def __init__(self, in_dim, res_size, out_dim, rho=0.9, leak=1.0):
        super().__init__()
        self.in_dim= in_dim
        self.res_size= res_size
        self.out_dim=out_dim
        self.rho= rho
        self.leak = leak

        self.W_in= nn.Parameter(torch.randn(res_size, in_dim) * 0.1)
        self.W_res= nn.Parameter(torch.randn(res_size, res_size) * 0.1)
        self.W_out= nn.Linear(res_size, out_dim)

        self.reset_state()

    def normalize_spectral_radius(self):
        _, largest_eig= torch.linalg.eig(self.W_res)
        self.W_res.data *= self.rho / largest_eig.abs()

    def forward(self, x):
        batch_size= x.size(0)
        combined_input= torch.cat((x.unsqueeze(2), self.res_state.unsqueeze(0).expand(batch_size, -1, -1)), dim=2)
        self.res_state= self.leak * (F.dropout(self.res_state, p=0.1) \ 
                                     + torch.matmul(self.W_in, x) \
                                     + torch.matmul(self.W_res, self.res_state.unsqueeze(0).expand(batch_size, -1, -1)))
        output=self.W_out(F.tanh(self.res_state.squeeze(2)))
        return output.squeeze(1)

    def reset_state(self):
        self.res_state= torch.zeros(self.res_size)
