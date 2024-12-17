'''
This file sets up neural layers: neurons and their output synapses
'''
import numpy as np
from synapsemodels.synapse_general import Synapse
from neuronmodels.neuron_general import Neuron


class layer():
    '''
    Interface class for all Layers.

    Functions as abstract class.
    Generates array of Neurons of the same type.
    Generates array of Synapses of the same type.
    Runs layer for one timestep.
    Generates function for accessing Synapse array.
    '''
    def __init__(self,
                layer_size: int,
                next_layer_size: int,
                neuron_type: Neuron,
                synapse_type: Synapse,
                input_synapses: np.ndarray):
        '''
        Generates instance of a layer

        layer_size: int; Neurons in this layer
        next_layer_size: int; Neurons in target layer
        neuron_type: Neuron; type of Neuron this layer will be constructed from
        synapse_type: Synapse; type of Synapse between this layer and target layer
        input_synapses: array; array of synapses from previous layer
        '''
        self.layer_size = layer_size
        self.next_layer_size = next_layer_size
        self.neuron_type = neuron_type
        self.synapse_type = synapse_type
        self._setup_neurons(input_synapses)

    @staticmethod
    def _setup_synapses(count, synapse_type: Synapse, rng: np.random.Generator) -> np.ndarray:
        syn_network = np.empty(count, dtype=Synapse)
        for synapse in range(count):
            syn_network[synapse] = synapse_type(synapse, rng)
        return syn_network
    
    def _setup_neurons(self, input_synapses: np.ndarray):
        network = np.empty(self.layer_size, dtype=Neuron)
        output_synapses = self._get_output_synapses()
        for neuron in range(self.layer_size):
            network[neuron] = self.neuron_type(neuron, input_synapses[neuron], output_synapses[neuron])
        self.neurons = network
    
    def step(self):
        '''
        Runs all Neurons then all Synapses for one timestep.
        '''
        for neuron in self.neurons:
            neuron.step()
        for synapse in self.synapses:
            synapse.step()

    def _get_output_synapses():
        raise NotImplementedError
    
    def get_input_synapses(): # To be used by the next layer. Returns array of arrays: [neurons[synapses]]
        raise NotImplementedError


class fully_connected_layer(layer):
    def __init__(self,
                layer_size: int,
                next_layer_size: int,
                neuron_type: Neuron,
                synapse_type: Synapse,
                input_synapses: np.ndarray,
                rng: np.random.Generator):
        '''
        Generates instance of a fully connected layer

        layer_size: int; Neurons in this layer
        next_layer_size: int; Neurons in target layer
        neuron_type: Neuron; type of Neuron this layer will be constructed from
        synapse_type: Synapse; type of Synapse between this layer and target layer
        input_synapses: array; array of synapses from previous layer
        rng: np.random.Generator; random number generator inherited by all synapses
        '''
        self.synapses = self._setup_synapses(layer_size * next_layer_size, synapse_type, rng)
        super().__init__(layer_size, next_layer_size, neuron_type, synapse_type, input_synapses)

    def _get_output_synapses(self) -> np.ndarray:
        synapses = np.empty(self.layer_size, dtype=Synapse)
        for neuron in range(self.layer_size):
            synapses[neuron] = np.array(self.synapses[[int(i + neuron*self.next_layer_size)
                                                       for i in range(self.next_layer_size)]],
                                        dtype=Synapse)
        return synapses
    
    def get_input_synapses(self) -> np.ndarray:
        '''
        Gets array of synapses organised by target neuron

        returns: array of arrays; [neuron[synapse]]
        '''
        synapses = np.empty(self.next_layer_size, dtype=Synapse)
        for neuron in range(self.next_layer_size):
            synapses[neuron] = np.array(self.synapses[[int(neuron + i*self.next_layer_size)
                                                       for i in range(self.layer_size)]],
                                        dtype=Synapse)
        return synapses

    
class final_layer(layer):
    def __init__(self, layer_size: int,
                 neuron_type: Neuron,
                 input_synapses: function):
        '''
        Generates instance of a final layer.

        This class generates one Interpretter for each Neuron (instead of Synapses).
        Functions as the end of a sequence of layers.

        layer_size: int; Neurons in this layer
        neuron_type: Neuron; type of Neuron this layer will be constructed from
        input_synapses: func; instance of get_input_synapses() from previous layer
        '''
        self.synapses = self._setup_synapses(neuron_type.interpretter)
        super().__init__(layer_size, None, neuron_type, None, input_synapses)
        
    def _setup_synapses(self, type): # Function which makes neuron of type interpretter
        interpretters = np.empty(self.layer_size, dtype= type)
        for i in range(self.layer_size):
            interpretters[i] = type(i)
        return interpretters

    def _get_output_synapses(self): # Function establishes 1:1 neuron to interpretter
        interpretters = np.empty(self.layer_size, dtype=self.neuron_type.interpretter)
        for neuron in range(self.layer_size):
            interpretters[neuron] = np.array(self.synapses[[neuron]], dtype=self.neuron_type.interpretter)
        return interpretters
    
