from synapsemodels import SpikeSyn
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=1)
synapse = SpikeSyn(rng, 1)
steps = int(1000/synapse.timestep) #Run for 1 second
nt = []
receptors = []
conductance = []
spikes = 0

print(f'nt_delta: {synapse.nt_delta}')
print(f'receptor_delta: {synapse.receptor_delta}')

for i in range(steps):
    synapse.run_step()
    if not(i % 100):
        synapse.spike()
        spikes += 1
    nt.append(synapse.c)
    receptors.append(synapse.d)
    conductance.append(synapse.g)


plt.plot(conductance, 'g')
plt.plot(nt, 'b')
plt.plot(receptors,'r')
plt.show()
