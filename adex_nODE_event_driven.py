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



if __name__ == "__main__":



    system = AdEx(V_rest=-0.060, event_driven=True)
    times, voltage, adapt, event_times = system.simulate()

    #system = AdEx(V_rest=-0.065, event_driven=True)
    #times, voltage2, adapt2, event_times = system.simulate()

    #loss = (voltage - voltage2).sum()
    #loss.backward()

    times = times.detach().cpu().numpy()
    voltage = voltage.detach().cpu().numpy() * 1000
    adapt = adapt.detach().cpu().numpy() * 1000
    

    plt.figure(figsize=(7, 3.5))

    

    vel, = plt.plot(times, adapt, color="C1", alpha=0.7, linestyle="--", linewidth=2.0)
    pos, = plt.plot(times, voltage, color="C0", linewidth=2.0)

    plt.hlines(0, 0, 100)
    plt.xlim([times[0], times[-1]])
    plt.ylim([voltage.min() - 20, 20])
    plt.ylabel("Membrane Voltage (mV)", fontsize=16)
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
