#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pyabf
from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event



torch.set_default_dtype(torch.float64)

#Does not seem to speed up if its on GPU
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#else:
device = torch.device('cpu')




class AdEx(nn.Module):

    def __init__(self, V_rest=-0.068, adjoint=False, event_driven=False, device=torch.device('cpu')):
        super().__init__()
        #Must be in SI units
        self.device = device
        self.V_rest = nn.Parameter(torch.as_tensor([V_rest]))
        self.V_reset = nn.Parameter(torch.as_tensor([-0.068]))
        self.V_T = nn.Parameter(torch.as_tensor([-0.045]))
        self.V_thres = nn.Parameter(torch.as_tensor([0.000]))
        self.delta_T = nn.Parameter(torch.as_tensor([0.010]))
        self.t0 = torch.tensor([0.0]).to(device)
        self.V_intial = nn.Parameter(torch.tensor([-0.068]))
        self.w_intial = nn.Parameter(torch.tensor([0.0]))
        self.R = nn.Parameter(torch.tensor([1000e6]))
        self.tau = nn.Parameter(torch.tensor([0.01]))
        self.tau_w = nn.Parameter(torch.tensor([0.02]))
        self.a = nn.Parameter(torch.tensor([0.09e-9]))
        self.b = nn.Parameter(torch.tensor([2e-12]))
        self.step_size = torch.tensor(5e-5).to(device)
        self.t = torch.as_tensor(np.arange(0, 2, 5e-5)).to(device)
        np_I = np.zeros(np.arange(0, 2, 5e-5).shape[0]+5)
        np_I[5561:11561] = -20e-12
        np_I[11561:25561] = 20e-12
        self.I_ext = torch.as_tensor(np_I).to(device)

        self.odeint = odeint_adjoint if adjoint else odeint
        self.forward = self._forward_no_reset if event_driven else self._forward_with_reset
        self.simulate = self._simulate_event if event_driven else self._simulate_full
        self.to(device)

    def _forward_with_reset(self, t, state):
        V, w = state
        ## Code adapted from https://github.com/PKU-NIP-Lab/BrainPy-Models
        ## Under GPL
        dvdt = (- (V - self.V_rest) + self.delta_T * torch.exp((V - self.V_T) / self.delta_T) - self.R * w + self.R * self.I_ext[int(t//self.step_size)]) / self.tau #nn.Parameter(torch.tensor([0.02]))
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
        dvdt = torch.where(V > self.V_thres, (self.V_reset-V)/self.step_size, dvdt)
        dwdt = torch.where(V > self.V_thres, dwdt + (self.b/self.step_size), dwdt)
        return dvdt, dwdt

    def _forward_no_reset(self, t, state):
        V, w = state
        ## Code adapted from https://github.com/PKU-NIP-Lab/BrainPy-Models
        ## Under GPL
        dvdt = (- (V - self.V_rest) + self.delta_T * torch.exp((V - self.V_T) / self.delta_T) - self.R * w + self.R * self.I_ext[int(t//self.step_size)]) / self.tau #nn.Parameter(torch.tensor([0.02]))
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
        return dvdt, dwdt

    def event_fn(self, t, state):
        V, w = state
        return -(V - self.V_thres)

    def get_initial_state(self):
        state = (self.V_intial, self.w_intial)
        return self.t0, state

    def state_update(self, state):
        V, w = state
        V = self.V_reset  
        w += self.b
        return (V, w)

    def get_collision_times(self, nspikes=1):

        event_times = []
        error = True
        t0, state = self.get_initial_state()

        while error:
            #This ensures we get finding events (spikes) until there are not more, but will likely break autograd
            # come up with a way to fix it?
            try:
                event_t, solution = odeint_event(self, state, t0, event_fn=self.event_fn, reverse_time=False, atol=5e-5, rtol=5e-5, odeint_interface=self.odeint, method='euler', options={'step_size':self.step_size})
                event_times.append(event_t)

                state = self.state_update(tuple(s[-1] for s in solution))
                t0 = event_t
            except: 
                error = False
        event_times.append(self.t[-1])
        return event_times

    def _simulate_full(self):
        t0, state = self.get_initial_state()
        solution = odeint(self, state, self.t, atol=1e-8, rtol=1e-8, method='euler', options={'step_size':self.step_size})
        voltage = solution[0]
        adapt = solution[1]
        return self.t, voltage.reshape(-1), adapt.reshape(-1)

    def _simulate_event(self, nspikes=1):
        
        # get dense path
        t0, state = self.get_initial_state()
        event_times = self.get_collision_times(nspikes=10)
        voltage = [state[0][None]]
        adapt = [state[1][None]]
        times = [t0.reshape(-1)]
        for event_t in event_times:
            tt = torch.linspace(float(t0), float(event_t), int((float(event_t) - float(t0)) * (1/self.step_size) + 1)).to(self.device)[1:-1]
            tt = torch.cat([t0.reshape(-1), tt, event_t.reshape(-1)])
            solution = odeint(self, state, tt, atol=5e-5, rtol=5e-5, method='euler', options={'step_size':self.step_size, 'perturb': False})

            voltage.append(solution[0])
            adapt.append(solution[1])
            times.append(tt)

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return torch.cat(times), torch.cat(voltage, dim=0).reshape(-1), torch.cat(adapt, dim=0).reshape(-1), torch.stack(event_times)