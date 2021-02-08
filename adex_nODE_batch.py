#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event



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
        self.V_T = nn.Parameter(torch.as_tensor([-0.04]))
        self.V_thres = nn.Parameter(torch.as_tensor([-0.010]))
        self.delta_T = nn.Parameter(torch.as_tensor([0.005]))
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.V_intial = nn.Parameter(torch.tensor([-0.068]))
        self.w_intial = nn.Parameter(torch.tensor([0.0]))
        self.R = nn.Parameter(torch.tensor([200e6]))
        self.tau = nn.Parameter(torch.tensor([0.01]))
        self.tau_w = nn.Parameter(torch.tensor([0.02]))
        self.a = nn.Parameter(torch.tensor([3.e-9]))
        self.b = nn.Parameter(torch.tensor([10e-12]))
        self.step_size = torch.tensor(0.0001)
        self.t = torch.as_tensor(np.arange(0, 1, 0.0001))
        np_I = np.zeros(np.arange(0, 1, 0.001).shape[0]+5)
        np_I[100:200] = -200e-12
        np_I[200:600] = 500e-12
        self.I_ext = torch.as_tensor(np_I)

        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, t, state):
        V, w = state
        
        dvdt = (- (V - self.V_rest) + self.delta_T * torch.exp((V - self.V_T) / self.delta_T) - self.R * w + self.R * self.I_ext[int(t//0.001)]) / self.tau #nn.Parameter(torch.tensor([0.02]))
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
        dvdt = torch.where(V > self.V_thres, (self.V_reset-V)/self.step_size, dvdt)
        dwdt = torch.where(V > self.V_thres, dwdt + (self.b/self.step_size), dwdt)
        return dvdt, dwdt

    def event_fn(self, t, state):
        # positive if ball in mid-air, negative if ball within ground.
        V, w = state
        return -(V - self.V_thres)

    def get_initial_state(self):
        state = (self.V_intial, self.w_intial)
        return self.t0, state

    def state_update(self, state):
        """ Updates state based on an event (collision)."""
        V, w = state
        V = self.V_reset  # need to add a small eps so as not to trigger the event function immediately.
        w += self.b
        return (V, w)

    def get_collision_times(self, nbounces=1):

        event_times = []

        t0, state = self.get_initial_state()

        for i in range(nbounces):
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




if __name__ == "__main__":

    system = AdEx().to(device)
    times, voltage, adapt = system.simulate()
    
    system = AdEx(V_rest=-0.08).to(device)

    plt.figure(figsize=(7, 3.5))

    for p in system.parameters():
        p.requires_grad = False
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
        

    vel, = plt.plot(times_, adapt_, color="C1", alpha=0.00001, linestyle="--", linewidth=2.0)
    pos, = plt.plot(times_, voltage_, color="C0", linewidth=2.0)
       # _, = plt.plot(times_, adapt2.detach().cpu().numpy(), color="r", alpha=0.7, linestyle="--", linewidth=2.0)
    _, = plt.plot(times_, voltage2.detach().cpu().numpy()*1000, color="r", linewidth=2.0)
    plt.hlines(0, 0, 100)
    plt.xlim([times[0], times[-1]])
    plt.ylim([-100, 20])
    plt.ylabel("Markov State", fontsize=16)
    plt.xlabel("Time", fontsize=13)
    plt.legend([pos, vel], ["Position", "adapt"], fontsize=16)

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
