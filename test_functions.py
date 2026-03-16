from abc import ABC, abstractmethod
import math
import numpy as np
import torch
import openmdao.api as om
from satellite_openmdao import satelliteGroup
from aerostructures_openmdao import aerostructuresGroup
from turbine_openmdao import turbineGroup

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

# Custom MDA problem example
class Satellite_modified(MDA):
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
        # return self._u21 - (x1*x4 + x4**2 + x5 + self._u12)
        return self._u21 - (x1*x4 + x4**2 + x5 + 9.0)

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


# Custom MDA problem example
class Satellite_modified_3dis(MDA):
    def __init__(self):
        # Design and coupling vars; problem specific
        self._x = torch.zeros(5)
        self._u12 = 0
        self._u21 = 0
        self._f = 0

        # Properties below are required for all problem classes.
        self._bounds = torch.tensor([[0., 0., 0., 0., 0., 6., 6.],
                                    [2., 2., 2., 2., 2., 12.,20.]])
        self._dim = 8
        self._input_dim = 5
        self._coupling_dim = 3
        self._tasks = [0, 1, 2]

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
        # return self._u21 - (x1*x4 + x4**2 + x5 + self._u12)
        return self._u21 - (x1*x4 + x4**2 + x5 + 9.0)

    @property
    def r3(self):
        x1, x2, x3, x4, x5 = [self._x[...,i] for i in range(self._input_dim)]
        g1 = 4.5 - (x1**2 + 2*x2 + x3 + x2*np.exp(-self._u21))
        g2 = x1**0.5 + x4 + x5*0.4*x1
        return self._f - (g2 - g1)

    @property
    def res(self):
        return torch.column_stack([self.r1, self.r2, self.r3])

    def set_vars(self, x) -> None:
        if x.shape[1] != self._dim:
            raise ValueError()
        self._x = x[:,:self._input_dim]
        self._u12 = x[:,-3]
        self._u21 = x[:,-2]
        self._f = x[:,-1]

    def set_bounds(self, bounds) -> None:
        if bounds.shape[1] != self._dim or bounds.shape[0] != 2:
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


# Custom MDA problem example
class Circuit(MDA):
    def __init__(self):
        # Design and coupling vars; problem specific
        self._x = torch.zeros(2) # [I_in, Vg]
        self._I1 = 0
        self._I2 = 0
        self._V1 = 0

        # Properties below are required for all problem classes.
        self._bounds = torch.tensor([[0., 0., 0., 0., 0.],
                                    [1., 10., 1., 1., 100.]])
        self._dim = 5
        self._input_dim = 2
        self._coupling_dim = 3
        self._tasks = [0, 1, 2]

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
        I_in, Vg = self._x[...,0], self._x[...,1]
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

# Custom MDA problem example
class Turbine(MDA):
    def __init__(self):
        # Design and coupling vars; problem specific
        # x = [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, TG, Fperf, Fecon]
        self._x = torch.zeros(11)
        self._Tbulk = 0
        self._tfail = 0
        self._Peng = 0
        self._recon = 0

        # Properties below are required for all problem classes.
        self._bounds = torch.tensor([[590, 610], # Tc1
                                     [640, 660], # Tc2
                                     [690, 710], # Tc3
                                     [29, 31], # K
                                     [1975, 2025], # hle
                                     [975, 1025], # hte
                                     [2.45e4, 2.55e4], # Plm
                                     [0.108, 0.132], # mdot
                                     [1225, 1275], # Tg
                                     [0.85, 0.95], # Fperf
                                     [0.9, 1.1], # Fecon
                                     [1000, 1200], # Tbulk
                                     [0.01, 100], # tfail
                                     [4.8e6, 6.6e6], # Peng
                                     [0, 1.0e5]], # recon
                                   ).transpose(0,1)
        
        self._dim = 15
        self._input_dim = 11
        self._coupling_dim = 4
        self._tasks = [0, 1, 2, 3]

        self._residual_mode = 'all'

        # set up MATLAB
        import turbineFEM
        import matlab
        self._double = matlab.double
        self._solveFEM = turbineFEM.initialize()
        self._geometry_filename = "turbine_blade.STL"

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
        Tc1, Tc2, Tc3 = self._x[...,0], self._x[...,1], self._x[...,2]
        K = self._x[...,3]
        hle, hte = self._x[...,4], self._x[...,5]

        # x_in = matlab.double([hle, hte, K, Tc1, Tc2, Tc3], size = (1,6))
        x_in = self._double(np.array(torch.column_stack([hle, hte, K, Tc1, Tc2, Tc3]), dtype='d'), size = (len(self._x),6))
                
        return self._Tbulk - torch.tensor(self._solveFEM.turbineFEM(self._geometry_filename, x_in)).squeeze()
        # return self._Tbulk - 0.5*(1200+650)

    @property
    def r2(self):
        Plm = self._x[...,6]

        # print(Plm)
        # print(self._tfail)
        # print(self._Tbulk)
        # print(np.log(self._tfail))
        # print(Plm/self._Tbulk - 20)

        # log transformed residual
        # print(min(np.log(self._tfail) - (Plm/self._Tbulk - 20)))
        # print(max(np.log(self._tfail) - (Plm/self._Tbulk - 20)))
        return np.log(self._tfail) - (Plm/self._Tbulk - 20)

    @property
    def r3(self):
        mdot, Tg, Fperf = self._x[...,7], self._x[...,8], self._x[...,9]

        mdot0 = 30
        T0 = 300
        N = 90
        Cp = 1003.5

        return self._Peng - (Fperf*(mdot0 - N*mdot)*Cp*T0*(1+Tg/T0-2*np.sqrt(Tg/T0)))

    @property
    def r4(self):
        Fecon = self._x[...,10]

        c0 = 0.07

        return self._recon - (Fecon*self._tfail*self._Peng*(c0/1000))

    def set_res_mode(self, mode):
        allowed_modes = ['all', 'diagonal']
        if mode in allowed_modes:
            self._residual_mode = mode
        return

    @property
    def res(self):
        if self._residual_mode == 'all':
            return torch.column_stack([self.r1, self.r2, self.r3, self.r4])
        # Workaround for slow res. Consider implementing a more universal solution.
        elif self._residual_mode == 'diagonal': 
            # Store current inputs
            x = torch.column_stack((self._x, self._Tbulk, self._tfail, self._Peng, self._recon))
            assert len(x) == self._coupling_dim

            # List of residual functions
            res_func_list = [Turbine.r1.fget, Turbine.r2.fget, Turbine.r3.fget, Turbine.r4.fget]
            res_out_list = []

            # Compute one residual for each input row
            for row, res_func in zip(x, res_func_list):
                self.set_vars(row.unsqueeze(0))
                res_out_list.append(res_func(self))

            # Set inputs back to what they were before
            self.set_vars(x)

            # Return the 4 residuals as a diagonal matrix
            return torch.diag(torch.tensor(res_out_list))

    def set_vars(self, x) -> None:
        if x.shape[1] != 15:
            raise ValueError()
        self._x = x[:,:11]
        self._Tbulk = x[:,-4]
        self._tfail = x[:,-3]
        self._Peng = x[:,-2]
        self._recon = x[:,-1]
        return

    def set_bounds(self, bounds) -> None:
        if bounds.shape[1] != 15 or bounds.shape[0] != 2:
            raise ValueError()
        self._bounds = bounds

    def _run_OpenMDAO(self, x_input):
        prob = om.Problem(turbineGroup())
        prob.setup()
        
        # prob.set_val('x', np.array([600,650,700,30,2000,1000,2.5e4,0.12,1250,0.9,1.0]))
        prob.set_val('x', x_input)
        
        prob.run_model()
        # print(prob['Tbulk'])
        # print(prob['tfail'])
        # print(prob['Peng'])
        # print(prob['recon'])

        self._openmdao_result = prob

    def from_OpenMDAO(self, x_input):
        self._run_OpenMDAO(x_input)
        prob = self._openmdao_result
        return torch.tensor([prob.get_val('Tbulk').item(), 
                             prob.get_val('tfail').item(),
                             prob.get_val('Peng').item(),
                             prob.get_val('recon').item()])

    def __del__(self):
        self._solveFEM.terminate()

class Turbine_modified(MDA):
    def __init__(self):
        # Design and coupling vars; problem specific
        # x = [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, TG, Fperf, Fecon]
        self._x = torch.zeros(11)
        self._Tbulk = 0
        self._tfail = 1

        # Properties below are required for all problem classes.
        self._bounds = torch.tensor([[590, 610], # Tc1
                                     [640, 660], # Tc2
                                     [690, 710], # Tc3
                                     [29, 31], # K
                                     [1975, 2025], # hle
                                     [975, 1025], # hte
                                     [2.45e+4, 2.55e+4], # Plm
                                     [0.108, 0.132], # mdot
                                     [1225, 1275], # TG
                                     [0.85, 0.95], # Fperf
                                     [0.9, 1.1], # Fecon
                                     [900, 1275], # Tbulk
                                     [1, 1000] # tfail
                                    ]).transpose(0,1)
        
        self._dim = 13
        self._input_dim = 11
        self._coupling_dim = 2
        self._tasks = [0, 1]

        # set up MATLAB
        import turbineFEM
        import matlab
        self._double = matlab.double
        self._solveFEM = turbineFEM.initialize()
        self._geometry_filename = "turbine_blade.STL"

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
        Tc1, Tc2, Tc3 = self._x[...,0], self._x[...,1], self._x[...,2]
        K = self._x[...,3]
        hle, hte = self._x[...,4], self._x[...,5]

        # x_in = matlab.double([hle, hte, K, Tc1, Tc2, Tc3], size = (1,6))
        x_in = self._double(np.array(torch.column_stack([hle, hte, K, Tc1, Tc2, Tc3]), dtype='d'), size = (len(self._x),6))
                
        return self._Tbulk - torch.tensor(self._solveFEM.turbineFEM(self._geometry_filename, x_in)).squeeze()
        # return self._Tbulk - 0.5*(1200+650)

    @property
    def r2(self):
        Plm = self._x[...,6]

        # log transformed residual
        return np.log(self._tfail) - (Plm/self._Tbulk - 20)

    @property
    def res(self):
        return torch.column_stack([self.r1, self.r2])

    def set_vars(self, x) -> None:
        if x.shape[1] != 13:
            raise ValueError()
        self._x = x[:,:11]
        self._Tbulk = x[:,-2]
        self._tfail = x[:,-1]

    def set_bounds(self, bounds) -> None:
        if bounds.shape[1] != 13 or bounds.shape[0] != 2:
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
        # prob = om.Problem()
        # prob.model = satelliteGroup()
        # prob.model.linear_solver = om.LinearBlockGS()
        
        # prob.driver = om.ScipyOptimizeDriver()
        # prob.driver.options['optimizer'] = 'COBYQA'
        # prob.driver.options['tol'] = 1e-8
        # prob.driver.options['disp'] = False
        
        # prob.model.add_design_var('x', lower=np.ones(5)*0, upper=np.ones(5)*2)
        # prob.model.add_objective('f')
        # prob.model.add_constraint('u12', lower=0.)
        
        # prob.model.set_input_defaults('x', np.ones(5))
        
        # prob.model.approx_totals()
        
        # prob.setup()
        
        # prob.set_val('x', x_input)
        
        # prob.run_model()
        
        # # return torch.tensor([prob.get_val('u12'), prob.get_val('u21')])
        # # return prob
        # self._openmdao_result = prob
        self._openmdao_result = 0

    def from_OpenMDAO(self, x_input):
        self._run_OpenMDAO(x_input)
        prob = self._openmdao_result
        # return torch.tensor([prob.get_val('u12').item(), prob.get_val('u21').item()])
        return 0

    def __del__(self):
        self._solveFEM.terminate()