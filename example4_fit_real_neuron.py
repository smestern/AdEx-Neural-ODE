#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pyabf
from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event
from utils import *


torch.set_default_dtype(torch.float64)

#Does not seem to speed up if its on GPU
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#else:
device = torch.device('cpu')




class AdEx(nn.Module):

    def __init__(self, V_rest=-0.068, adjoint=False):
        super().__init__()
        #Must be in SI units
        self.V_rest = nn.Parameter(torch.as_tensor([V_rest]))
        self.V_reset = nn.Parameter(torch.as_tensor([-0.068]))
        self.V_T = nn.Parameter(torch.as_tensor([-0.045]))
        self.V_thres = nn.Parameter(torch.as_tensor([0.020]))
        self.delta_T = nn.Parameter(torch.as_tensor([0.010]))
        self.t0 = torch.tensor([0.0])
        self.V_intial = nn.Parameter(torch.tensor([-0.068]))
        self.w_intial = nn.Parameter(torch.tensor([0.0]))
        self.R = nn.Parameter(torch.tensor([1.8e9]))
        self.tau = nn.Parameter(torch.tensor([0.025]))
        self.tau_w = nn.Parameter(torch.tensor([0.02]))
        self.a = nn.Parameter(torch.tensor([0.09e-9]))
        self.b = nn.Parameter(torch.tensor([2e-12]))
        self.step_size = torch.tensor(5e-5) 
        self.t = torch.as_tensor(np.arange(0, 2, 5e-5))
        self.t.requires_grad = True
        np_I = np.zeros(np.arange(0, 2, 5e-5).shape[0]+5)
        np_I[5561:11561] = -20e-12
        np_I[11561:25561] = 20e-12
        self.I_ext = torch.as_tensor(np_I)
        self.spike_times = torch.as_tensor([])
        self.spike_times.requires_grad = True
        self.threshold = torch.as_tensor([-0.01])
        self.threshold.requires_grad = True
        self.odeint = odeint_adjoint if adjoint else odeint
        

    def forward(self, t, state):
        V, w = state
        
        dvdt = (- (V - self.V_rest) + self.delta_T * torch.exp((V - self.V_T) / self.delta_T) - self.R * w + self.R * self.I_ext[int(t//5e-5)]) / self.tau #nn.Parameter(torch.tensor([0.02]))
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
        dvdt = torch.where(V > self.V_thres, (self.V_reset-V)/self.step_size, dvdt)
        dwdt = torch.where(V > self.V_thres, dwdt + (self.b/self.step_size), dwdt)
        if V > self.V_thres:
            self.spike_times = torch.hstack((self.spike_times, t))
        return dvdt, dwdt

    def event_fn(self, t, state):
        
        V, w = state
        return -(V - self.V_thres)

    def get_initial_state(self):
        state = (self.V_intial, self.w_intial)
        self.spike_times = torch.as_tensor([])
        self.spike_times.requires_grad = True
        return self.t0, state

    def find_spikes(self, x, y, threshold):
        spike_bin = torch.nn(y>-0.01, 1., 2.)#torch.where(y>=threshold,1, 0)
        spike_idx = torch.nonzero((spike_bin[1:] - spike_bin[:-1]))[::2].reshape(-1)
        return torch.take(x, spike_idx)

    def state_update(self, state):
        """ Updates state based on an event (collision)."""
        V, w = state
        V = self.V_reset  
        w += self.b
        return (V, w)

    def get_collision_times(self, nspikes=1):

        event_times = []

        t0, state = self.get_initial_state()

        for i in range(nspikes):
            event_t, solution = odeint_event(self, state, t0, event_fn=self.event_fn, reverse_time=False, atol=1e-8, rtol=1e-8, odeint_interface=self.odeint, method='euler', options={'step_size':self.step_size})
            event_times.append(event_t)

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return event_times

    def simulate(self):
        t0, state = self.get_initial_state()
        solution = odeint(self, state, self.t, atol=1e-8, rtol=1e-8, method='euler', options={'step_size':self.step_size})
        voltage = solution[0]
        adapt = solution[1]
        return self.t, voltage.reshape(-1), adapt.reshape(-1)

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
    system = AdEx(V_rest=-0.068).to(device)

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
