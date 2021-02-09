#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from adex_node import AdEx
from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event



torch.set_default_dtype(torch.float64)

#Does not seem to speed up if its on GPU
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#else:
device = torch.device('cpu')

if __name__ == "__main__":

    system = AdEx().to(device)
    times, voltage, adapt = system.simulate()
    
    system = AdEx(V_rest=-0.08).to(device)

    plt.figure(figsize=(7, 3.5))

    #for p in system.parameters():
     #   p.requires_grad = False
    system.V_rest.requires_grad =True
    optim = torch.optim.Adam(system.parameters(), lr=5e-4)
    for epoch in np.arange(10):
        optim.zero_grad()
        times, voltage2, adapt2 = system.simulate()
        
        loss = (voltage - voltage2).sum()
        loss.backward(retain_graph=True)
        
        optim.step() #gradient descent
        print(loss)
        print(system.V_rest)
    times_ = times.detach().cpu().numpy()
    voltage_ = voltage.detach().cpu().numpy() * 1000
    adapt_ = adapt.detach().cpu().numpy() * 1000
        

        
    plt.clf()
        

    volt2, = plt.plot(times_, adapt_, color="C1", alpha=0.00001, linestyle="--", linewidth=2.0)
    volt, = plt.plot(times_, voltage_, color="C0", linewidth=2.0)
       # _, = plt.plot(times_, adapt2.detach().cpu().numpy(), color="r", alpha=0.7, linestyle="--", linewidth=2.0)
    _, = plt.plot(times_, voltage2.detach().cpu().numpy()*1000, color="r", linewidth=2.0)
    plt.hlines(0, 0, 100)
    plt.xlim([times[0], times[-1]])
    plt.ylim([-100, 20])
    plt.ylabel("Membrane Voltage (mV)", fontsize=16)
    plt.xlabel("Time", fontsize=13)
    plt.legend([volt, volt2], ["Fit 1", "adapt"], fontsize=16)

    plt.gca().xaxis.set_tick_params(direction='in', which='both')  # The bottom will maintain the default of 'out'
    plt.gca().yaxis.set_tick_params(direction='in', which='both')  # The bottom will maintain the default of 'out'

        # Hide the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    plt.tight_layout()
        #plt.pause(0.05)
    plt.savefig(f"{epoch}_bouncing_ball.png")
