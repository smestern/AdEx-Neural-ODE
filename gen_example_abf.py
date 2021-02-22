import numpy as np
import pyabf
from ipfx import feature_extractor as sp

if __name__=="__main__":
    abf = pyabf.ABF("example.abf")
    #get sweep 5
    abf.setSweep(4)
    dataX, dataY, dataC = abf.sweepX, abf.sweepY, abf.sweepC
    dt = 1/abf.dataRate
    print(f"dt of {dt}")
    #get the stimulus indexes
    stim_idx = np.nonzero(np.diff(dataC))[0]
    #record these for later
    print(f"Stim changes at idx {stim_idx}")
    #Also get the values
    stim_val = np.unique(dataC)
    print(f"stim values of {stim_val}")
    #save the response
    #up to 2 seconds
    end_idx = int(2//dt) + 1
    np.save("example.npy", dataY[:end_idx])

    #get the spike times
    sp_ex = sp.SpikeFeatureExtractor(filter=0)
    spikes = sp_ex.process( dataX, dataY, dataC)
    spike_times = spikes['threshold_t'].to_numpy()
    np.save("example_spikes.npy", spike_times)
