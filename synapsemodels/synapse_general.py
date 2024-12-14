'''
Parent class of synapse
Contains all shared parameters
Taken from Isbister et al 2018
https://pmc.ncbi.nlm.nih.gov/articles/PMC6015810/table/RSFS20180021TB1/
'''
import numpy as np

class Synapse:
    conductance_decay = 150 #150
    nt_decay = 25 #Array of values used 5, 25, 125
    nt_concentration = 0.5
    receptor_decay = 25 #Array of values used 5, 25, 125
    receptor_unblocked = 0.5
    weight_decay = 10 #Not given in original publication
    learning_rate = 0.1
    scaling_factor = 0.1 #Various values used (and somewhat unclear)
    timestep = 0.2

    nt_delta = np.exp((-timestep / nt_decay))
    receptor_delta = np.exp((-timestep / receptor_decay))
    conductance_delta  = np.exp((-timestep / conductance_decay))

    def __init__(self, id, rng):
        self.id = id
        self.weight = rng.random()
        self.g = 0
        self.c = 0
        self.d = 0

    @classmethod
    def set_time(cls, timestep):
        cls.timestep = timestep

    def step(self):
        raise NotImplementedError(f"Run_step function not implemented for {type(self)}")
        
    def spike(self):
        raise NotImplementedError(f"Spike function not implemented for {type(self)}")
    
    def backprop(self):
        raise NotImplementedError(f'Backprop function not implemented for {type(self)}')

        

