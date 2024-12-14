from spiking_neuron import SpikeNeuron
import matplotlib.pyplot as plt
import numpy as np

neuron = SpikeNeuron(1, [], [])
steps = 1000/neuron.timestep #equal to 1 second
input_current = np.array([0.12]) # 0.75 = current for max bio spike rate (200Hz) .12 give min 12Hz (.119 gives 0)
membrane_potential = []
true_vals = 0


for i in range(steps):
    input_current = np.random.uniform(0.05,0.2)
    neuron.step(input_current)
    membrane_potential.append(neuron.mem_pot)
    if neuron.spiked:
        membrane_potential.append(0)
    true_vals += neuron.spiked


print(f'frequency: {true_vals}')
#plt.plot(membrane_potential)
#plt.show()