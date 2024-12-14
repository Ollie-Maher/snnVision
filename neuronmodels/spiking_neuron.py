'''
This file establishes the spiking neuron model
'''
import numpy as np
from .neuron_general import Neuron 


class SpikeNeuron(Neuron):
    
    def __init__ (self, id: int, input_synapses: np.ndarray, output_synapses: np.ndarray):
        super().__init__(id, input_synapses, output_synapses)
        self.mem_pot = self.resting_pot
        self.spiked = False
        self.period = int(self.refractory_period/self.timestep)
        self.spiketime = 0
    
    def _input_conductance(self):
        inputs = [x.g for x in self.input_synapses]
        self.conductance = np.sum(inputs) * (self.reversal_pot - self.mem_pot)

    def _potential(self):
        self.mem_pot = (self.mem_pot - self.resting_pot) * self.mem_pot_delta + self.resting_pot  + (self.resistance * self.conductance)
        if self.mem_pot > self.threshold:
            self._spike()

    def _spike(self):
        self.mem_pot = self.resting_pot
        self.spiked = True

        for synapse in self.output_synapses:
            synapse.spike()
        for synpase in self.input_synapses:
            synapse.backprop()

    def step(self):
        self._input_conductance()
        if self.spiked:
            self.spiked = False
            self.spiketime = 0
        elif self.spiketime >= self.period:
            self._potential()
        elif self.spiketime < self.period:
            self.spiketime += 1

            

        

        