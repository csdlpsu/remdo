from abc import ABC, abstractmethod
import math
import numpy as np
import torch
import openmdao.api as om
from satellite_openmdao import satelliteGroup

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

# Custom MDA problem example
class Satellite(MDA):
    def __init__(self):
        self.x = torch.zeros(5)
        self.u12 = 0
        self.u21 = 0
        self.bounds = torch.tensor([[0., 0., 0., 0., 0., 6., 6.],
                                    [2., 2., 2., 2., 2., 12.,20.]])
        self.dim = 7
        self.input_dim = 5
        self.coupling_dim = 2
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

    # @property
    # def g1(self):
    #     x1, x2, x3 = self.x[...,0], self.x[...,1], self.x[...,2]
    #     return 4.5 - (x1**2 + 2*x2 + x3 + x2*np.exp(-self.u21))

    # @property
    # def g2(self):
    #     x1, x4, x5 = self.x[...,0], self.x[...,3], self.x[...,4]
    #     return x1**0.5 + x4 + x5*0.4*x1

    # @property
    # def obj(self):
    #     return self.g1 + self.g2

    def _runOpenMDAO(self, x_input):
        prob = om.Problem()
        prob.model = satelliteGroup()
        prob.model.linear_solver = om.LinearBlockGS()
        
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYQA'
        prob.driver.options['tol'] = 1e-8
        prob.driver.options['disp'] = False
        
        prob.model.add_design_var('x', lower=np.ones(5)*0, upper=np.ones(5)*2)
        prob.model.add_objective('f')
        prob.model.add_constraint('u12', lower=0.)
        
        prob.model.set_input_defaults('x', np.ones(5))
        
        prob.model.approx_totals()
        
        prob.setup()
        
        prob.set_val('x', x_input)
        
        prob.run_model()
        
        # return torch.tensor([prob.get_val('u12'), prob.get_val('u21')])
        # return prob
        self.openmdao_result = prob

    def fromOpenMDAO(self, x_input):
        self._runOpenMDAO(x_input)
        prob = self.openmdao_result
        return torch.tensor([prob.get_val('u12').item(), prob.get_val('u21').item()])
