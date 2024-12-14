from layerTypes import fully_connected_layer
from neuronmodels import SpikeNeuron
from synapsemodels import SpikeSyn
from numpy import random
from matplotlib import pyplot
import numpy as np

layer_size = 5
hidden_size = 5
rng = random.default_rng(seed=1)
steps = 5000 #Run for 1 second at 0.2ms per step

fc_layer = fully_connected_layer(layer_size, hidden_size, SpikeNeuron, SpikeSyn, rng)
fc_layer2 = fully_connected_layer(hidden_size, 1, SpikeNeuron, SpikeSyn, rng)

in_syn_shell = np.empty(hidden_size, dtype=SpikeSyn)
for i in range(5):
    in_syn_shell[i] = np.array([SpikeSyn(i, rng)], dtype=SpikeSyn)

fc_layer.setup_neurons(in_syn_shell)
fc_layer2.setup_neurons(fc_layer.get_input_synapses())

outputs = dict(
    Neuron1 = [],
    Neuron2 = [],
    Neuron3 = [],
    Neuron4 = [],
    Neuron5 = []
)

for step in range(steps):
    for syn in range(layer_size):
        if rng.uniform() > 0.999:
            in_syn_shell[syn][0].spike()
    fc_layer.step()
    fc_layer2.step()
    for count, neuron in enumerate(fc_layer2.neurons):
        outputs[f'Neuron{count+1}'].append(neuron.mem_pot)


for key in outputs.keys():
    pyplot.plot(outputs[key])
pyplot.show()

