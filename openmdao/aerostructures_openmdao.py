import numpy as np
import openmdao.api as om
import os
import torch

from remdo.config import as_tensor, tensor, to_numpy, zeros
from remdo.openmdao_loader import load_openmdao_symbol

MDA = load_openmdao_symbol("problem_base.py", "MDA")

os.environ['OPENMDAO_REPORTS'] = 'none'
os.environ['OPENMDAO_USE_MPI'] = 'false'

class aerodynamicsDis(om.ExplicitComponent):
    """Aerodynamics discipline mapping beam parameter and twist to lift."""

    def setup(self):
        """Declare aerodynamics inputs and lift output."""

        # Global Design Variable
        self.add_input('B', val=0.)

        # Coupling parameter
        self.add_input('phi', val=0.)

        # Coupling output
        self.add_output('L', val=0.)

    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate the aerodynamic lift equation."""

        q = 1 # N/cm2
        C = 10 # cm
        psi = 0.05 # rad
        r = 0.9425
        theta0 = 0.26 # rad

        B = inputs['B']
        phi = inputs['phi']

        outputs['L'] = 1/1000 * q*B*C * ((2*np.pi*(phi+psi)) + r*(1-np.cos(np.pi/2*(phi+psi)/theta0)))

class structuresDis(om.ExplicitComponent):
    """Structural discipline mapping lift to twist angle."""

    def setup(self):
        """Declare structural input and twist output."""

        # Coupling parameter
        self.add_input('L', val=0.)

        # Coupling output
        self.add_output('phi', val=0.)

    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate the structural deflection equation."""

        C = 10 # cm
        p = 0.1111
        k1 = 4000 # N/cm
        k2 = 2000 # N/cm
        z1 = 0.2
        z2 = 0.7

        L = inputs['L']

        outputs['phi'] = np.remainder(( 1000*L/(k1*(1+p)) - (1000*L*p)/(k2*(1+p)) ) * ( 1/(C*(z2-z1)) ), 2*np.pi)
        
class aerostructuresGroup(om.Group):
    """OpenMDAO group for the coupled aerostructures benchmark."""

    def setup(self):
        """Assemble aerodynamic and structural disciplines with a block solver."""

        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('aero', aerodynamicsDis(), promotes_inputs=['B', 'phi'],
                            promotes_outputs=['L'])
        cycle.add_subsystem('strux', structuresDis(), promotes_inputs=['L'],
                            promotes_outputs=['phi'])

        cycle.linear_solver = om.DirectSolver(rhs_checking=False)
        nlbgs = cycle.nonlinear_solver = om.NonlinearBlockGS()
        nlbgs.options['maxiter'] = 1000
        nlbgs.options['iprint'] = 0
        

class Aerostructures(MDA):
    """Two-discipline aero-structural fixed-point benchmark."""

    def __init__(self):
        self._B = zeros(1)
        self._L = zeros(1)
        self._phi = zeros(1)
        self._bounds = tensor([[0.0, -20.0, -torch.pi / 2.0], [300.0, 20.0, torch.pi / 2.0]])
        self._dim = 3
        self._input_dim = 1
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
        """Aerodynamic lift residual."""

        q = 1.0
        C = 10.0
        psi = 0.05
        r = 0.9425
        theta0 = 0.26
        return self._L - 1.0 / 1000.0 * q * self._B * C * (
            (2.0 * torch.pi * (self._phi + psi))
            + r * (1.0 - torch.cos(torch.pi / 2.0 * (self._phi + psi) / theta0))
        )

    @property
    def r2(self) -> torch.Tensor:
        """Structural deflection residual."""

        C = 10.0
        p = 0.1111
        k1 = 4000.0
        k2 = 2000.0
        z1 = 0.2
        z2 = 0.7
        return self._phi - (1000.0 * self._L / (k1 * (1.0 + p)) - (1000.0 * self._L * p) / (k2 * (1.0 + p))) * (
            1.0 / (C * (z2 - z1))
        )

    @property
    def res(self) -> torch.Tensor:
        return torch.column_stack([self.r1, self.r2])

    def set_vars(self, x: torch.Tensor) -> None:
        x = as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected {self.dim} variables, got {x.shape[1]}.")
        self._B = x[:, 0]
        self._L = x[:, 1]
        self._phi = x[:, 2]

    def _run_OpenMDAO(self, x_input: torch.Tensor):
        """Run the OpenMDAO aerostructures model."""

        prob = om.Problem()
        prob.model = aerostructuresGroup()
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "COBYQA"
        prob.driver.options["tol"] = 1e-8
        prob.driver.options["disp"] = False
        prob.model.add_design_var("B", lower=0.0, upper=300.0)
        prob.model.approx_totals()
        prob.setup()
        prob.set_val("B", to_numpy(as_tensor(x_input)))
        prob.run_model()
        return prob

    def from_OpenMDAO(self, x_input: torch.Tensor) -> torch.Tensor:
        prob = self._run_OpenMDAO(x_input)
        return tensor([prob.get_val("L").item(), prob.get_val("phi").item()])
