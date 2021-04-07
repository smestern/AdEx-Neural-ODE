import numpy as np
import torch
import matplotlib.pyplot as plt


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
    def __call__(self, x, y, spikes_, res_spikes):
        #mse = torch.mean(torch.square(x - y))
        #max_se = torch.square(x.max() - y.max())
        #min_se = torch.square(x.min() - y.min())

        spikes_new = torch.nn.utils.rnn.pad_sequence((spikes_, res_spikes))

        spike_mse = torch.sum(torch.abs(spikes_new[:, 0] - spikes_new[:,1]))
        return  spike_mse #(mse + max_se + min_se + spike_mse)

class EMD():
    def __init__(self):
        pass
    def __call__(self, x, y):
        #mse = torch.mean(torch.square(x - y))
        #max_se = torch.square(x.max() - y.max())
        #min_se = torch.square(x.min() - y.min())
        x = torch.nn.functional.softmax(x)
        y = torch.nn.functional.softmax(y)
        error = self.torch_cdf_loss(x, y)
        return  error

    def torch_cdf_loss(self, tensor_a,tensor_b):
        # last-dimension is weight distribution
        # p is the norm of the distance, p=1 --> First Wasserstein Distance
        # to get a positive weight with our normalized distribution
        # we recommend combining this loss with other difference-based losses like L1

        # normalize distribution, add 1e-14 to divisor to avoid 0/0
        tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
        tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
        # make cdf with cumsum
        cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
        cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

        # choose different formulas for different norm situations
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
        cdf_loss = cdf_distance.mean()
        return cdf_loss


class EMDT():
    def __init__(self):
        pass
    def __call__(self, x, y, threshold=-0.02):
        #mse = torch.mean(torch.square(x - y))
        #max_se = torch.square(x.max() - y.max())
        #min_se = torch.square(x.min() - y.min())

        x_t = torch.nn.functional.threshold(x, threshold, threshold) - threshold
        y_t = torch.nn.functional.threshold(y, threshold, threshold) - threshold
        #x_n = torch.nn.functional.softmax(x_t)
       #y_n = torch.nn.functional.softmax(y_t)
        
        error = self.torch_cdf_loss(x_t, y_t)
        return  error

    def torch_cdf_loss(self, tensor_a,tensor_b):
        # last-dimension is weight distribution
        # p is the norm of the distance, p=1 --> First Wasserstein Distance
        # to get a positive weight with our normalized distribution
        # we recommend combining this loss with other difference-based losses like L1

        # normalize distribution, add 1e-14 to divisor to avoid 0/0
        tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
        tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
        # make cdf with cumsum
        cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
        cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

        # choose different formulas for different norm situations
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
        cdf_loss = cdf_distance.mean()
        return cdf_loss        

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])