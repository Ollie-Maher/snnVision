'''
This file loads visual data for the basic interpreter
'''
from keras.datasets import mnist


class mnist_data:
    def __init__(self, rng):
        (self.train_imgs, self.train_labels), (self.test_imgs, self.test_labels) = mnist.load_data()
        self.id = 0
        self.rng = rng
        
    def get_next(self, img: bool = True, lbl: bool = True, test_set: bool = False) -> dict:
        return_set = dict(
            image = None,
            label = None
        )
        if img:
            return_set['image'] = getattr(self, f'{"test" if test_set else "train"}_imgs')
        if lbl:
            return_set['label'] = getattr(self, f'{"test" if test_set else "train"}_labels')
        self.id += 1
        return return_set
    
    def get_set(self, count, **kwargs):
        sample = []
        for i in range(count):
            sample.append(self.get_next(**kwargs))

