'''
general neuron parameters and functions
'''
import numpy as np

class Neuron:
    '''
    Interface class for all neurons.

    Functions as an abstract class. Contains key class variables.
    All neurons have a step function and an interpretter inner class
    '''
    mempot_decay = 20
    resting_pot = -0.074
    threshold = -0.050
    resistance = 0.04
    refractory_period = 20
    reversal_pot = 0
    timestep = 0.2

    mem_pot_delta = np.exp((-timestep / mempot_decay))

    def __init__(self, id: int, input_synapses: np.ndarray, output_synapses: np.ndarray):
        self.id = id
        self.input_synapses = input_synapses
        self.output_synapses = output_synapses

    @classmethod
    def set_time(cls, timestep):
        '''
        Sets timestep for all neurons of this class.

        timestep: time period of each processing step in milliseconds.
        '''
        cls.timestep = timestep

    def step(self):
        '''
        Runs parameters for one timestep.
        '''
        raise NotImplementedError(f"Step function not implemented for {type(self)}")

    class interpretter(): # Class of Synapse interface for outputting data from neurons.
        '''
        Inner class of a Neuron.

        This class is required for extracting and interpretting data from a Neuron.
        Virtual subclass of Synapse superclass. 
        '''
        def __init__(self):
            raise NotImplementedError(f"interpretter function not implemented for {type(self)}")
        
        def step(self):
            '''
            Processes one timestep of output
            '''
            raise NotImplementedError(f"interpretter step function not implemented for {type(self)}")
        
        def spike(self):
            '''
            Processes spike when called
            '''
            raise NotImplementedError(f"interpretter spike function not implemented for {type(self)}")