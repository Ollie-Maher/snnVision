from neuronmodels import SpikeNeuron
from synapsemodels import SpikeSyn
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=1)
neuron = SpikeNeuron(1, [1], [1])
synapse = SpikeSyn(rng, 1)
steps = int(1000/neuron.timestep) #Run for 1 second
membrane_potential = []
conductance = []
nt = []
receptors = []
weight = []
spikes = 0

for i in range(steps):
    synapse.run_step()
    if rng.uniform() > 0.995:
        synapse.spike()
        spikes += 1
    neuron.step(synapse.g)
    if neuron.spiked:
        synapse.backprop()
    membrane_potential.append(neuron.mem_pot)
    nt.append(synapse.c)
    receptors.append(synapse.d)
    conductance.append(synapse.g)

print(spikes)
plt.plot(conductance, 'g')
plt.plot(nt, 'b')
plt.plot(receptors, 'r')
plt.show()
plt.plot(membrane_potential)
plt.show()

