'''
This file established the class for spiking synpases
'''

import numpy as np
from .synapse_general import Synapse


class SpikeSyn(Synapse):
    

    def __init__(self, id, rng):
        super().__init__(id, rng)
    
    def _transmitter(self):
        self.c = self.c * self.nt_delta

    def _receptor(self):
        self.d = self.d * self.receptor_delta

    def _conductance(self):
        self.g = self.g * self.conductance_delta

    def spike(self):
        self.c += self.nt_concentration * (1 - self.c)
        self.g += self.scaling_factor * self.weight
        self.weight -= self.weight * self.d * self.learning_rate

    def backprop(self):
        self.d += self.receptor_unblocked * (1 - self.d)
        self.weight += (1-self.weight) * self.c * self.learning_rate

    def step(self):
        self._transmitter()
        self._receptor()
        self._conductance()

    