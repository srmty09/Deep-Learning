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
    


