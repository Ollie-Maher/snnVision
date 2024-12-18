'''
This file loads visual data for the basic interpreter
'''
from keras import mnist


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
        img_sampler = getattr(mnist_data, f'{"test" if test_set else "train"}_imgs')
        label_sampler = getattr(mnist_data, f'{"test" if test_set else "train"}_labels')
        if img:
            return_set['image'] = img_sampler(self, self.id)
        if lbl:
            return_set['label'] = label_sampler(self, self.id)
        self.id += 1
        return return_set

a = mnist_data()

print(a.get_next())
print(a.get_next(test_set=True))
