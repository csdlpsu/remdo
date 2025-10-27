from abc import ABC, abstractmethod
import math
import numpy as np
import torch

class MDA(ABC):
    # @property
    # @abstractmethod
    # def bounds(self):
    #     pass

    # @property
    # @abstractmethod
    # def tasks(self):
    #     pass
        
    @property
    @abstractmethod
    def res(self):
        pass

    @abstractmethod
    def set_vars(self):
        pass

# class MDAO(MDA):
#     @property
#     @abstractmethod
#     def obj(self):
#         pass

# Custom MDA problem example
class Satellite(MDA):
    def __init__(self):
        self.x = torch.zeros(5)
        self.u12 = 0
        self.u21 = 0
        self.bounds = torch.tensor([[0., 0., 0., 0., 0., 6., 6.],
                                    [2., 2., 2., 2., 2., 12.,20.]])
        self.tasks = [0, 1]

    def set_vars(self, x) -> None:
        if x.shape[1] != 7:
            raise ValueError()
        self.x = x[:,:5]
        self.u12 = x[:,-2]
        self.u21 = x[:,-1]

    def set_bounds(self, bounds) -> None:
        if x.shape[1] != 7 or x.shape[0] != 2:
            raise ValueError()
        self.bounds = bounds

    @property
    def r1(self):
        x1, x2, x3 = self.x[...,0], self.x[...,1], self.x[...,2]
        return self.u12 - (x1**2 + 2*x2 - x3 + 2*self.u21**0.5)

    @property
    def r2(self):
        x1, x4, x5 = self.x[...,0], self.x[...,3], self.x[...,4]
        return self.u21 - (x1*x4 + x4**2 + x5 + self.u12)

    @property
    def res(self):
        return torch.column_stack([self.r1, self.r2])

    @property
    def g1(self):
        x1, x2, x3 = self.x[...,0], self.x[...,1], self.x[...,2]
        return 4.5 - (x1**2 + 2*x2 + x3 + x2*np.exp(-self.u21))

    @property
    def g2(self):
        x1, x4, x5 = self.x[...,0], self.x[...,3], self.x[...,4]
        return x1**0.5 + x4 + x5*0.4*x1

    @property
    def obj(self):
        return self.g1 + self.g2