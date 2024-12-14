from neuronmodels import SpikeNeuron
from synapsemodels import SpikeSyn
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=1)
neuron1 = SpikeNeuron(1, [1], [1])
neuron2 = SpikeNeuron(2, [1], [1])
synapse = SpikeSyn(rng, 1)
steps = int(1000/0.2) #Run for 1 second
membrane_potential = []
conductance = []
nt = []
receptors = []
weight = []
inspikes = 0
outspikes = 0

print(synapse.weight)

for i in range(steps):
    neuron1.step(0.13)
    if neuron1.spiked:
        synapse.spike()
        inspikes += 1
    synapse.run_step()
    neuron2.step(synapse.g)
    if neuron2.spiked:
        synapse.backprop()
        outspikes += 1
    membrane_potential.append(neuron2.mem_pot)
    nt.append(synapse.c)
    receptors.append(synapse.d)
    conductance.append(synapse.g)

print(synapse.weight)
print(inspikes, outspikes)
plt.plot(conductance, 'g')
plt.plot(nt, 'b')
plt.plot(receptors, 'r')
plt.show()
plt.plot(membrane_potential)
plt.show()

