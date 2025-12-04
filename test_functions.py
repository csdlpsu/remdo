from abc import ABC, abstractmethod
import math
import numpy as np
import torch
import openmdao.api as om
from satellite_openmdao import satelliteGroup
from aerostructures_openmdao import aerostructuresGroup

# Abstract base class for MDA problems. 
class MDA(ABC):
    @property
    @abstractmethod
    def bounds(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass
    
    @property
    @abstractmethod
    def input_dim(self):
        pass

    @property
    @abstractmethod
    def coupling_dim(self):
        pass

    @property
    @abstractmethod
    def tasks(self):
        pass
        
    @property
    @abstractmethod
    def res(self):
        pass

    @abstractmethod
    def set_vars(self):
        pass

    @abstractmethod
    def set_bounds(self):
        pass

    @abstractmethod
    def from_OpenMDAO(self, x_input):
        pass


# Custom MDA problem example
class Satellite(MDA):
    def __init__(self):
        # Design and coupling vars; problem specific
        self._x = torch.zeros(5)
        self._u12 = 0
        self._u21 = 0

        # Properties below are required for all problem classes.
        self._bounds = torch.tensor([[0., 0., 0., 0., 0., 6., 6.],
                                    [2., 2., 2., 2., 2., 12.,20.]])
        self._dim = 7
        self._input_dim = 5
        self._coupling_dim = 2
        self._tasks = [0, 1]

    @property
    def bounds(self):
        return self._bounds

    @property
    def dim(self):
        return self._dim

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def coupling_dim(self):
        return self._coupling_dim

    @property
    def tasks(self):
        return self._tasks

    @property
    def r1(self):
        x1, x2, x3 = self._x[...,0], self._x[...,1], self._x[...,2]
        return self._u12 - (x1**2 + 2*x2 - x3 + 2*self._u21**0.5)

    @property
    def r2(self):
        x1, x4, x5 = self._x[...,0], self._x[...,3], self._x[...,4]
        return self._u21 - (x1*x4 + x4**2 + x5 + self._u12)

    @property
    def res(self):
        return torch.column_stack([self.r1, self.r2])

    def set_vars(self, x) -> None:
        if x.shape[1] != 7:
            raise ValueError()
        self._x = x[:,:5]
        self._u12 = x[:,-2]
        self._u21 = x[:,-1]

    def set_bounds(self, bounds) -> None:
        if bounds.shape[1] != 7 or bounds.shape[0] != 2:
            raise ValueError()
        self._bounds = bounds

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

    def _run_OpenMDAO(self, x_input):
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
        self._openmdao_result = prob

    def from_OpenMDAO(self, x_input):
        self._run_OpenMDAO(x_input)
        prob = self._openmdao_result
        return torch.tensor([prob.get_val('u12').item(), prob.get_val('u21').item()])



# Aerostructural problem example from Ghoreishi and Imani (2020)
class Aerostructures(MDA):
    def __init__(self):
        # Design and coupling vars; problem specific
        # self._q = 0
        # self._C = 0
        # self._psi = 0
        # self._r = 0
        # self._theta0 = 0
        # self._p = 0
        # self._k = torch.zeros(2)
        # self._z = torch.zeros(2)
        self._B = 0

        # # Combined vector of design vars
        # self._X = torch.hstack(self._q, self._C, self_.psi, self._r, self._theta0, self._p, self._k, self._z, self._B)
        
        self._L = 0
        self._phi = 0

        # Properties below are required for all problem classes.
        self._bounds = torch.tensor([[0,   -20,   -np.pi/2],
                                     [300, 20, np.pi/2]])
        self._dim = 3
        self._input_dim = 1
        self._coupling_dim = 2
        self._tasks = [0, 1]

    @property
    def bounds(self):
        return self._bounds

    @property
    def dim(self):
        return self._dim

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def coupling_dim(self):
        return self._coupling_dim

    @property
    def tasks(self):
        return self._tasks

    @property
    def r1(self):
        q = 1 # N/cm2
        C = 10 # cm
        psi = 0.05 # rad
        r = 0.9425
        theta0 = 0.26 # rad
        
        B = self._B
        phi = self._phi
        L = self._L
        
        return L - 1/1000*q*B*C*((2*np.pi*(phi+psi)) + r*(1-torch.cos(np.pi/2*(phi+psi)/theta0))) # PLEASE CHECK THIS
        # Checked

    @property
    def r2(self):
        C = 10 # cm
        p = 0.1111
        k1 = 4000 # N/cm
        k2 = 2000 # N/cm
        z1 = 0.2
        z2 = 0.7

        B = self._B
        L = self._L
        phi = self._phi

        return phi - (1000*L/(k1*(1+p))-(1000*L*p)/(k2*(1+p)))*(1/(C*(z2-z1)))
        
    @property
    def res(self):
        return torch.column_stack([self.r1, self.r2])

    def set_vars(self, x) -> None:
        if x.shape[1] != 3:
            raise ValueError()
        self._B = x[:,0]
        self._L = x[:,1]
        self._phi = x[:,2]

    def set_bounds(self, bounds) -> None:
        if x.shape[1] != 3 or x.shape[0] != 2:
            raise ValueError()
        self._bounds = bounds

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

    def _run_OpenMDAO(self, x_input):
        prob = om.Problem()
        prob.model = aerostructuresGroup()
        
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYQA'
        prob.driver.options['tol'] = 1e-8
        prob.driver.options['disp'] = False
        
        prob.model.add_design_var('B', lower = 0., upper = 300.)
        
        prob.model.approx_totals()
        
        prob.setup()
        
        prob.set_val('B', x_input)
        
        prob.run_model()
        
        self._openmdao_result = prob

    def from_OpenMDAO(self, x_input):
        self._run_OpenMDAO(x_input)
        prob = self._openmdao_result
        return torch.tensor([prob.get_val('L').item(), prob.get_val('phi').item()])
