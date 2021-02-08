#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event

torch.set_default_dtype(torch.float64)


class AdEx(nn.Module):

    def __init__(self, V_rest=-0.068, adjoint=False):
        super().__init__()
        #Must be in SI units
        self.V_rest = nn.Parameter(torch.as_tensor([V_rest]))
        self.V_reset = nn.Parameter(torch.as_tensor([-0.068]))
        self.V_T = nn.Parameter(torch.as_tensor([-0.04]))
        self.V_thres = nn.Parameter(torch.as_tensor([0.010]))
        self.delta_T = nn.Parameter(torch.as_tensor([0.001]))
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.V_intial = nn.Parameter(torch.tensor([-0.068]))
        self.w_intial = nn.Parameter(torch.tensor([0.0]))
        self.R = nn.Parameter(torch.tensor([200e6]))
        self.tau = nn.Parameter(torch.tensor([0.01]))
        self.tau_w = nn.Parameter(torch.tensor([0.02]))
        self.a = nn.Parameter(torch.tensor([3.e-9]))
        self.b = nn.Parameter(torch.tensor([10e-12]))
        self.t = torch.as_tensor(np.arange(0, 1, 0.00001))
        np_I = np.zeros(np.arange(0, 1, 0.001).shape[0])
        np_I[100:200] = -200e-12
        np_I[200:600] = 500e-12
        self.I_ext = torch.as_tensor(np_I)

        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, t, state):
        V, w = state
        dvdt = (- (V - self.V_rest) + self.delta_T * torch.exp((V - self.V_T) / self.delta_T) - self.R * w + self.R * self.I_ext[int(t//0.001)]) / self.tau #nn.Parameter(torch.tensor([0.02]))
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
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
        error = True
        t0, state = self.get_initial_state()

        while error:
            try:
                event_t, solution = odeint_event(self, state, t0, event_fn=self.event_fn, reverse_time=False, atol=1e-8, rtol=1e-8, odeint_interface=self.odeint, method='euler', options={'step_size':0.00001})
                event_times.append(event_t)

                state = self.state_update(tuple(s[-1] for s in solution))
                t0 = event_t
            except: 
                error = False
        event_times.append(self.t[-1])
        return event_times

    def simulate(self, nbounces=1):
        
        # get dense path
        t0, state = self.get_initial_state()
        event_times = self.get_collision_times(nbounces=10)
        voltage = [state[0][None]]
        adapt = [state[1][None]]
        times = [t0.reshape(-1)]
        for event_t in event_times:
            tt = torch.linspace(float(t0), float(event_t), int((float(event_t) - float(t0)) * 50000))[1:-1]
            tt = torch.cat([t0.reshape(-1), tt, event_t.reshape(-1)])
            solution = odeint(self, state, tt, atol=1e-8, rtol=1e-8, method='euler', options={'step_size':0.0001})

            voltage.append(solution[0])
            adapt.append(solution[1])
            times.append(tt)

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        #solution = odeint(self, state, self.t, method='euler', atol=1e-8, rtol=1e-8)

        #voltage = solution[0]
        #adapt = solution[1]
        

       

        return torch.cat(times), torch.cat(voltage, dim=0).reshape(-1), torch.cat(adapt, dim=0).reshape(-1), event_times




if __name__ == "__main__":



    system = AdEx()
    times, voltage, adapt, event_times = system.simulate()

    system = AdEx(V_rest=-0.065)
    times, voltage2, adapt2, event_times = system.simulate()

    loss = (voltage - voltage2).sum()
    loss.backward()

    times = times.detach().cpu().numpy()
    voltage = voltage.detach().cpu().numpy() * 1000
    adapt = adapt.detach().cpu().numpy() * 1000
    

    plt.figure(figsize=(7, 3.5))

    

    vel, = plt.plot(times, adapt, color="C1", alpha=0.7, linestyle="--", linewidth=2.0)
    pos, = plt.plot(times, voltage, color="C0", linewidth=2.0)

    plt.hlines(0, 0, 100)
    plt.xlim([times[0], times[-1]])
    plt.ylim([voltage.min() - 20, 20])
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
    plt.show()
    plt.savefig("bouncing_ball.png")
