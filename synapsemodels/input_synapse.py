'''
This file will received inputs and transform them for neural input
'''
from .synapse_general import Synapse


class input_syn(Synapse):
    max_rate = 200
    time_period = 1000/max_rate
    fixed_weight = 1
    coefficient = 5

    def __init__(self, id):
        self.id = id
        self.t_since_last = self.time_period
        self.g = 0
        self.spike_conductance = self.scaling_factor * self.fixed_weight
    
    def step(self, input: int):
        self.g = self.g * self.conductance_delta
        if input:
            self.spike(input)

    def spike(self, input):
        if self.t_since_last >= (self.coefficient/input) + self.time_period:
            self.g = self.spike_conductance # Conductance set to 1
            self.t_since_last
        else:
            self.t_since_last += 1

    def backprop(self):
        pass