import torch
import math

class Softmax():
    def __init__(self):
        pass
    
    def __call__(self, vec):
        for i in range(vec.size(0)):
            row = vec[i]

            _max = -float('inf')
            norm_fac = 0.0

            for j in range(row.size(0)):
                ele = row[j].item()
                if ele > _max:
                    correction_factor = math.exp(_max - ele)
                    norm_fac = norm_fac * correction_factor + math.exp(0.0)
                    _max = ele
                else:
                    norm_fac += math.exp(ele - _max)

            for j in range(row.size(0)):
                row[j] = math.exp(row[j].item() - _max) / norm_fac

        return vec


vec = torch.tensor([[3,2,5,1]],dtype=torch.float32)
softmax = Softmax()
vec = softmax(vec)
print(vec)