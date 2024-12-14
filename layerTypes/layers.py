'''
This file sets up neural layers: neurons and their output synapses
'''
import numpy as np
from synapsemodels.synapse_general import Synapse
from neuronmodels.neuron_general import Neuron


class layer():
    def __init__(self,
                layer_size: int,
                next_layer_size: int,
                neuron_type: Neuron,
                synapse_type: Synapse):
        self.layer_size = layer_size
        self.next_layer_size = next_layer_size
        self.neuron_type = neuron_type
        self.synapse_type = synapse_type

    @staticmethod
    def _setup_synapses(count, synapse_type: Synapse, rng: np.random.Generator) -> np.ndarray:
        syn_network = np.empty(count, dtype=Synapse)
        for synapse in range(count):
            syn_network[synapse] = synapse_type(synapse, rng)
        return syn_network
    
    def setup_neurons():
        raise NotImplementedError
    
    def _get_output_synapses():
        raise NotImplementedError
    
    def get_input_synapses():
        raise NotImplementedError

    def step():
        raise NotImplementedError


class fully_connected_layer(layer):
    def __init__(self,
                layer_size: int,
                next_layer_size: int,
                neuron_type: Neuron,
                synapse_type: Synapse,
                rng: np.random.Generator):
        super().__init__(layer_size, next_layer_size, neuron_type, synapse_type)
        self.synapses = self._setup_synapses(layer_size * next_layer_size, synapse_type, rng)

    def setup_neurons(self, input_synapses: np.ndarray):
        network = np.empty(self.layer_size, dtype=Neuron)
        output_synapses = self._get_output_synapses()
        for neuron in range(self.layer_size):
            network[neuron] = self.neuron_type(neuron, input_synapses[neuron], output_synapses[neuron])
        self.neurons = network
    
    def _get_output_synapses(self) -> np.ndarray:
        synapses = np.empty(self.layer_size, dtype=Synapse)
        for neuron in range(self.layer_size):
            synapses[neuron] = np.array(self.synapses[[int(i + neuron*self.next_layer_size)
                                                       for i in range(self.next_layer_size)]],
                                        dtype=Synapse)
        return synapses
    
    def get_input_synapses(self) -> np.ndarray: # To be used by the next layer
        synapses = np.empty(self.next_layer_size, dtype=Synapse)
        for neuron in range(self.next_layer_size):
            synapses[neuron] = np.array(self.synapses[[int(neuron+i*self.next_layer_size)
                                                       for i in range(self.layer_size)]],
                                        dtype=Synapse)
        return synapses

    def step(self):
        for neuron in self.neurons:
            neuron.step()
        for synapse in self.synapses:
            synapse.step()

    
    



