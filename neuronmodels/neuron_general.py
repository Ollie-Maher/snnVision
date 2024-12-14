'''
general neuron parameters and functions
'''
import numpy as np

class Neuron:
    mempot_decay = 20
    resting_pot = -0.074
    threshold = -0.050
    resistance = 0.04
    refractory_period = 2
    reversal_pot = 0
    timestep = 0.2

    mem_pot_delta = np.exp((-timestep / mempot_decay))

    def __init__(self, id: int, input_synapses: np.ndarray, output_synapses: np.ndarray):
        self.id = id
        self.input_synapses = input_synapses
        self.output_synapses = output_synapses

    @classmethod
    def set_time(cls, timestep):
        cls.timestep = timestep

    def step(self):
        raise NotImplementedError(f"Step function not implemented for {type(self)}")

