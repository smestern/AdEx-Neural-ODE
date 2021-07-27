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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if __name__ == "__main__":

    with torch.no_grad():
        res = torch.as_tensor(np.load("example.npy")/ 1000) 
        res_spikes = torch.as_tensor(np.load("example_spikes.npy"))

    system = AdEx(V_rest=-0.060, event_driven=True, adjoint=True)
    times, voltage, adapt, event_times = system.simulate()

    system2 = AdEx(V_rest=-0.065, event_driven=True).to(device)
    times2, voltage2, adapt2, event_times2 = system2.simulate()

    loss = nn.MSELoss()

    optim = torch.optim.Adam([{
        'params': [system.b, system.a, system.w_intial], 'lr': 1e-13},
        {'params': [system.R], 'lr': 1e7}, #Carefully define the learning rate for each parameter. 
        #Otherwise too high LR with explode the gradient, too low makes no difference on the param
        {'params': [system.V_rest, system.V_reset, system.V_T, system.delta_T, system.V_intial, system.tau, system.tau_w]}], lr=1e-4)
        
    with torch.no_grad():     
            print(system.__dict__)
    for x in np.arange(500):
        optim.zero_grad()
        times, voltage, adapt, event_times = system.simulate()
        out = loss(event_times[0], event_times2[0]) * 1000
        out += loss(event_times[-1], event_times2[-1]) * 1000
        out += loss(voltage[:40000], voltage2[:40000])
        out.backward(retain_graph=True)
        optim.step() #gradient descent
        print("==== loss ====")
        print(out)
        with torch.no_grad():     
            print(system.__dict__)
        #loss = (voltage - voltage2).sum()
            times_ = times.detach().cpu().numpy()
            times_2 = times2.detach().cpu().numpy()
            plt.clf()
            volt, = plt.plot(times_, voltage.detach().cpu().numpy() * 1000, color="C0", linewidth=2.0)
            volt2, = plt.plot(times_2, voltage2.detach().cpu().numpy()*1000, color="r", linewidth=2.0)
            fspikes = np.ravel(event_times.detach().cpu().numpy())
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
            plt.savefig(f"output/{x}_fit.png")




