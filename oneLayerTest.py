from layerTypes import fully_connected_layer, final_layer
from neuronmodels import SpikeNeuron
from synapsemodels import SpikeSyn
from numpy import random
from matplotlib import pyplot
import numpy as np

layer_size = 5
hidden_size = 5
rng = random.default_rng(seed=100)
steps = 5000 #Run for 1 second at 0.2ms per step

in_syn_shell = np.empty(layer_size, dtype=SpikeSyn)
for i in range(layer_size):
    in_syn_shell[i] = np.array([SpikeSyn(i, rng)], dtype=SpikeSyn)

fc_layer = fully_connected_layer(layer_size, hidden_size, SpikeNeuron, SpikeSyn, in_syn_shell, rng)
fc_copy = final_layer(layer_size, SpikeNeuron, in_syn_shell)
final_layer = final_layer(hidden_size, SpikeNeuron, fc_layer.get_input_synapses())

badneuron = fc_layer.neurons
spikes_fc = dict(
    Neuron1 = [],
    Neuron2 = [],
    Neuron3 = [],
    Neuron4 = [],
    Neuron5 = []
)
spikes_final = dict(
    Neuron1 = [],
    Neuron2 = [],
    Neuron3 = [],
    Neuron4 = [],
    Neuron5 = []
)
in_spikes = 0
mempots = np.empty((5,5000))
conductances = np.empty((5,5000))

for step in range(steps):
    for syn in range(layer_size):
        in_syn_shell[syn][0].step()
        if rng.uniform() > 0.995:
            in_syn_shell[syn][0].spike()
            in_spikes += 1    
    fc_layer.step()
    fc_copy.step()
    final_layer.step()
    for i, neuron in enumerate(badneuron):
        mempots[i, step] = neuron.mem_pot
    for i, synapse in enumerate(in_syn_shell):
        conductances[i, step] = synapse[0].g
    for count, interpretter in enumerate(fc_copy.synapses):
        if interpretter.spiking:
            spikes_fc[f'Neuron{count+1}'].append(step)
    for count, interpretter in enumerate(final_layer.synapses):
        if interpretter.spiking:
            spikes_final[f'Neuron{count+1}'].append(step)

print(in_spikes)

# Plotting results in raster plot
# set different colors for each neuron
colors = [f'C{i}' for i in range(len(spikes_final))]

# set different line properties for each set of positions
# note that some overlap
lineoffsets1 =  [i + 1 for i in range(len(spikes_final))]

fig, axs = pyplot.subplots(2)

# create plots
for i, key in enumerate(spikes_fc.keys()):
    axs[0].eventplot(spikes_fc[key], colors=colors[i], lineoffsets=lineoffsets1[i], linewidths = 1.5)
    #print(len(spikes_fc[key]))

for i, key in enumerate(spikes_final.keys()):
    axs[1].eventplot(spikes_final[key], colors=colors[i], lineoffsets=lineoffsets1[i], linewidths = 1.5)
    #print(len(spikes_final[key]))

pyplot.show()

_, axs = pyplot.subplots(2)

for i in range(mempots.shape[0]):
    axs[0].plot(conductances[i], color = colors[i])
    axs[1].plot(mempots[i], color = colors[i])

pyplot.show()

