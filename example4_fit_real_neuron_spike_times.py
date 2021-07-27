#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pyabf
from adex_node import AdEx
from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event
from utils import *


torch.set_default_dtype(torch.float64)

#Does not seem to speed up if its on GPU
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#else:
device = torch.device('cpu')



#Parameter constraints = 
constraint = [[-0.08, -0.040], #v_rest
        [-0.080, -0.02], #V_reset
        [-0.060, -0.045], #V_T
        [0.01, 0.02], #V_thres
        [0.001, 0.02], #d_t
        [-0.09, -0.03], #V_intial
        [0.0001e-13, 0.1e-13],
        [1.3e9, 2.6e9],
        [0.001, 0.1],
        [0.001, 0.1],
        [1e-12, 1e-9],
        [0.1e-13, 40e-12],
        ]




if __name__ == "__main__":
    with torch.no_grad():
        res = torch.as_tensor(np.load("example.npy")/ 1000) 
        res_spikes = torch.as_tensor(np.load("example_spikes.npy"))
    system = AdEx(V_rest=-0.06, event_driven=True).to(device)

    plt.figure(figsize=(7, 3.5))
    errorCalc = EMDT()
    constraintMod = paramConstraint(constraint)
    threshold = torch.tensor([-0.01])
    optim = torch.optim.Adam([{
        'params': [system.b, system.a, system.w_intial], 'lr': 1e-13},
        {'params': [system.R], 'lr': 1e7}, #Carefully define the learning rate for each parameter. 
        #Otherwise too high LR with explode the gradient, too low makes no difference on the param
        {'params': [system.V_rest, system.V_reset, system.V_T, system.delta_T, system.V_intial, system.tau, system.tau_w]}], lr=1e-4)
    for epoch in np.arange(500):
        print(f"==== Epoch {epoch} start ====")
        optim.zero_grad()
        system.apply(constraintMod)
        times, voltage2, adapt2 = system.simulate()
        #spike_times = find_spikes(times, voltage2, threshold)
        loss = errorCalc(res, voltage2)
        #loss_spikes.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        #plot_grad_flow(system.named_parameters())
        optim.step() #gradient descent
        print("==== loss ====")
        print(loss)
        print("==== PARAMETERS ====")
        with torch.no_grad():
            if epoch % 10 == 0:
                torch.save(system, f"checkpoints//{epoch}_model.pkl")
            
            print(system.__dict__)
            
            times_ = times.detach().cpu().numpy()
            
            plt.clf()
            volt, = plt.plot(times_, res.detach().cpu().numpy() * 1000, color="C0", linewidth=2.0)
            volt2, = plt.plot(times_, voltage2.detach().cpu().numpy()*1000, color="r", linewidth=2.0)
            fspikes = np.ravel(system.spike_times.detach().cpu().numpy())
            plt.scatter(fspikes, np.full(fspikes.shape[0], 0))
            plt.xlim([times[0], times[-1]])
            plt.ylim([-100, 20])
            plt.ylabel("Membrane Voltage (mV)", fontsize=16)
            plt.xlabel("Time", fontsize=13)
            plt.legend([volt, volt2], ["Real Trace", "Fit"], fontsize=16)

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
            plt.savefig(f"output/{epoch}_fit.png")
