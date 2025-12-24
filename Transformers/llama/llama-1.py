import torch
import torch.nn as nn



class SwiGLU(nn.Module):
    def __init__(self,dimension) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)
        

    def forward(self,x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swish = swish * self.linear_2(x)

        return swish


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))  

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        x_norm = x / (rms + self.eps)
        return x_norm * self.gamma
