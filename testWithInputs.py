from layerTypes import fully_connected_layer, final_layer
from neuronmodels import SpikeNeuron
from synapsemodels import SpikeSyn, input_syn
from inputs import mnist_data
from numpy import random
from matplotlib import pyplot
import numpy as np

layer_size = 28*28 # image size
hidden_size = 5
rng = random.default_rng(seed=100)
steps = 5000 #Run for 1 second at 0.2ms per step#

data = mnist_data(rng)
sample = data.get_next(lbl=False)
image = sample['image'].squeeze()
print(image, image.shape)

in_syn_shell = np.empty(layer_size, dtype=input_syn)
for i in range(layer_size):
    in_syn_shell[i] = np.array([input_syn(i)], dtype=input_syn)

fc_layer = fully_connected_layer(layer_size, hidden_size, SpikeNeuron, SpikeSyn, in_syn_shell, rng)
fc_copy = final_layer(layer_size, SpikeNeuron, in_syn_shell)
final_layer = final_layer(hidden_size, SpikeNeuron, fc_layer.get_input_synapses())

spikes_final = dict(
    Neuron1 = [],
    Neuron2 = [],
    Neuron3 = [],
    Neuron4 = [],
    Neuron5 = []
)

for step in range(steps):
    for syn in range(layer_size):
        in_syn_shell[syn][0].step(image[syn])  
    fc_layer.step()
    fc_copy.step()
    final_layer.step()
    for count, interpretter in enumerate(final_layer.synapses):
        if interpretter.spiking:
            spikes_final[f'Neuron{count+1}'].append(step)