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
    def __call__(self, x, y, spikes_, res_spikes):
        #mse = torch.mean(torch.square(x - y))
        #max_se = torch.square(x.max() - y.max())
        #min_se = torch.square(x.min() - y.min())

        spikes_new = torch.nn.utils.rnn.pad_sequence((spikes_, res_spikes))

        spike_mse = torch.mean(torch.abs(spikes_new[:, 0] - spikes_new[:,1]))
        return  spike_mse * 1000 #(mse + max_se + min_se + spike_mse)

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