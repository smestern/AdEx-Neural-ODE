import numpy as np
import torch

class paramConstraint():
    def __init__(self, constraint_ar):
        self.constraints = constraint_ar
    
    def __call__(self, module):
        for i, p in enumerate(module._parameters):
            
            module._parameters[p].data = torch.clamp(module._parameters[p], self.constraints[i][0], self.constraints[i][1])

class modifiedMSE():
    def __init__(self):
        pass
    def __call__(self, x, y):
        mse = torch.mean(torch.square(x - y))
        max_se = torch.square(x.max() - y.max())
        min_se = torch.square(x.min() - y.min())
        return (mse + max_se + min_se) * 1000

class modifiedMSE_with_spikes():
    def __init__(self):
        pass
    def __call__(self, x, y):
        mse = torch.mean(torch.square(x - y))
        max_se = torch.square(x.max() - y.max())
        min_se = torch.square(x.min() - y.min())

        x_crossings

        return (mse + max_se + min_se) * 1000