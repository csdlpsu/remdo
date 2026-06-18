import numpy as np
import openmdao.api as om
import os
import torch

from remdo.config import as_tensor, tensor, to_numpy, zeros
from remdo.openmdao_loader import load_openmdao_symbol

MDA = load_openmdao_symbol("problem_base.py", "MDA")

os.environ['OPENMDAO_REPORTS'] = 'none'
os.environ['OPENMDAO_USE_MPI'] = 'false'

class satelliteDis1(om.ExplicitComponent):
    """First satellite discipline mapping ``(x, u21)`` to ``u12``."""

    def setup(self):
        """Declare discipline inputs and coupling output."""

        # Global Design Variable
        self.add_input('x', val=np.ones(5))
        
        # Coupling parameter
        self.add_input('u21', val=9.)

        # Coupling output
        self.add_output('u12', val=9.)

    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate the first satellite discipline equation."""

        x1 = inputs['x'][0]
        x2 = inputs['x'][1]
        x3 = inputs['x'][2]
        u21 = inputs['u21']

        if u21.real < 0.0:
            u21 *= -1

        outputs['u12'] = x1**2 + 2*x2 - x3 + 2*(u21**0.5)

class satelliteDis2(om.ExplicitComponent):
    """Second satellite discipline mapping ``(x, u12)`` to ``u21``."""

    def setup(self):
        """Declare discipline inputs and coupling output."""

        # Global Design Variable
        self.add_input('x', val=np.ones(5))
        
        # Coupling parameter
        self.add_input('u12', val=9.)

        # Coupling output
        self.add_output('u21', val=9.)
        
    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate the second satellite discipline equation."""

        x1 = inputs['x'][0]
        x4 = inputs['x'][3]
        x5 = inputs['x'][4]
        u12 = inputs['u12']

        outputs['u21'] = x1*x4 + x4**2 + x5 + u12
        

class satelliteGroup(om.Group):
    """OpenMDAO group for the coupled two-discipline satellite benchmark."""

    def setup(self):
        """Assemble disciplines, nonlinear solver, and objective component."""

        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('d1', satelliteDis1(), promotes_inputs=['x', 'u21'], 
                           promotes_outputs=['u12'])
        cycle.add_subsystem('d2', satelliteDis2(), promotes_inputs=['x', 'u12'], 
                           promotes_outputs=['u21'])

        nlbgs = cycle.nonlinear_solver = om.NonlinearBlockGS()
        cycle.linear_solver = om.DirectSolver(rhs_checking=True)
        # nlbgs = cycle.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2)
        nlbgs.options['maxiter'] = 1000
        nlbgs.options['iprint'] = 0

        self.add_subsystem('obj_cmp', om.ExecComp('f = ((x[0]**0.5 + x[3] + 0.4*x[0]*x[4]) - (4.5 - (x[0]**2 + 2*x[1] + x[2] + x[1]*exp(-u21))))', x=np.ones(5), u21=9.0),
                           promotes_inputs=['x','u21'], promotes_outputs=['f'])
        # self.add_subsystem('res1', om.ExecComp('r1 = u12 - (x[0]**2 + 2*x[1] - x[2] + 2*(u21**0.5))', x=np.ones(5)*0.5, u21=9.0),
        #                    promotes_inputs=['x','u21'], promotes_outputs=['r1'])
        # self.add_subsystem('res2', om.ExecComp('r2 = u21 - (x[0]*x[3] + x[3]**2 + x[4] + u12)', x=np.ones(5), u12=9.0),
        #                    promotes_inputs=['x','u12'], promotes_outputs=['r2'])    


class Satellite(MDA):
    """Two-discipline analytical satellite benchmark problem.

    Variables are ordered as ``[x1, x2, x3, x4, x5, u12, u21]``.  Residuals
    enforce the two discipline coupling equations.
    """

    def __init__(self):
        self._x = zeros(1, 5)
        self._u12 = zeros(1)
        self._u21 = zeros(1)
        self._bounds = tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0], [2.0, 2.0, 2.0, 2.0, 2.0, 12.0, 20.0]])
        self._dim = 7
        self._input_dim = 5
        self._coupling_dim = 2
        self._tasks = [0, 1]

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def coupling_dim(self) -> int:
        return self._coupling_dim

    @property
    def tasks(self) -> list[int]:
        return self._tasks

    @property
    def r1(self) -> torch.Tensor:
        """Residual for discipline 1 output ``u12``."""

        x1, x2, x3 = self._x[..., 0], self._x[..., 1], self._x[..., 2]
        return self._u12 - (x1**2 + 2.0 * x2 - x3 + 2.0 * torch.sqrt(self._u21))

    @property
    def r2(self) -> torch.Tensor:
        """Residual for discipline 2 output ``u21``."""

        x1, x4, x5 = self._x[..., 0], self._x[..., 3], self._x[..., 4]
        return self._u21 - (x1 * x4 + x4**2 + x5 + self._u12)

    @property
    def res(self) -> torch.Tensor:
        return torch.column_stack([self.r1, self.r2])

    def set_vars(self, x: torch.Tensor) -> None:
        x = as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected {self.dim} variables, got {x.shape[1]}.")
        self._x = x[:, : self.input_dim]
        self._u12 = x[:, -2]
        self._u21 = x[:, -1]

    def _run_OpenMDAO(self, x_input: torch.Tensor):
        """Run the OpenMDAO satellite model for a fixed input vector."""

        prob = om.Problem()
        prob.model = satelliteGroup()
        prob.model.linear_solver = om.LinearBlockGS()
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "COBYQA"
        prob.driver.options["tol"] = 1e-8
        prob.driver.options["disp"] = False
        prob.model.add_design_var("x", lower=np.zeros(5), upper=np.ones(5) * 2.0)
        prob.model.add_objective("f")
        prob.model.add_constraint("u12", lower=0.0)
        prob.model.set_input_defaults("x", np.ones(5))
        prob.model.approx_totals()
        prob.setup()
        prob.set_val("x", to_numpy(as_tensor(x_input)))
        prob.run_model()
        return prob

    def from_OpenMDAO(self, x_input: torch.Tensor) -> torch.Tensor:
        prob = self._run_OpenMDAO(x_input)
        return tensor([prob.get_val("u12").item(), prob.get_val("u21").item()])


class SatelliteDirect(Satellite):
    """Direct satellite discipline map returning outputs instead of residuals."""

    @property
    def f1(self) -> torch.Tensor:
        """Discipline 1 output map."""

        x1, x2, x3 = self._x[..., 0], self._x[..., 1], self._x[..., 2]
        return x1**2 + 2.0 * x2 - x3 + 2.0 * torch.sqrt(self._u21)

    @property
    def f2(self) -> torch.Tensor:
        """Discipline 2 output map."""

        x1, x4, x5 = self._x[..., 0], self._x[..., 3], self._x[..., 4]
        return x1 * x4 + x4**2 + x5 + self._u12

    @property
    def res(self) -> torch.Tensor:
        return torch.column_stack([self.f1, self.f2])


class SatelliteModified(Satellite):
    """Satellite variant where the second residual uses fixed feed-forward coupling."""

    @property
    def r2(self) -> torch.Tensor:
        x1, x4, x5 = self._x[..., 0], self._x[..., 3], self._x[..., 4]
        return self._u21 - (x1 * x4 + x4**2 + x5 + 9.0)


class SatelliteModified3Dis(SatelliteModified):
    """Three-residual satellite variant that includes the objective residual."""

    def __init__(self):
        super().__init__()
        self._f = zeros(1)
        self._bounds = tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0, -10.0], [2.0, 2.0, 2.0, 2.0, 2.0, 12.0, 20.0, 10.0]])
        self._dim = 8
        self._coupling_dim = 3
        self._tasks = [0, 1, 2]

    @property
    def r3(self) -> torch.Tensor:
        """Residual for the satellite objective-like response."""

        x1, x2, x3, x4, x5 = [self._x[..., i] for i in range(self.input_dim)]
        g1 = 4.5 - (x1**2 + 2.0 * x2 + x3 + x2 * torch.exp(-self._u21))
        g2 = torch.sqrt(x1) + x4 + x5 * 0.4 * x1
        return self._f - (g2 - g1)

    @property
    def res(self) -> torch.Tensor:
        return torch.column_stack([self.r1, self.r2, self.r3])

    def set_vars(self, x: torch.Tensor) -> None:
        x = as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected {self.dim} variables, got {x.shape[1]}.")
        self._x = x[:, : self.input_dim]
        self._u12 = x[:, -3]
        self._u21 = x[:, -2]
        self._f = x[:, -1]

    def from_OpenMDAO(self, x_input: torch.Tensor) -> torch.Tensor:
        """Return OpenMDAO coupling variables and objective response."""

        prob = self._run_OpenMDAO(x_input)
        return tensor([prob.get_val("u12").item(), prob.get_val("u21").item(), prob.get_val("f").item()])


Satellite_direct = SatelliteDirect
Satellite_modified = SatelliteModified
Satellite_modified_3dis = SatelliteModified3Dis
