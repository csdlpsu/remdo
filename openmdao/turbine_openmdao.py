import numpy as np
import openmdao.api as om
import os
from pathlib import Path
import sys

import torch

from remdo.config import as_tensor, tensor, to_numpy, zeros
from remdo.openmdao_loader import load_openmdao_symbol

MDA = load_openmdao_symbol("problem_base.py", "MDA")

os.environ['OPENMDAO_REPORTS'] = 'none'
os.environ['OPENMDAO_USE_MPI'] = 'false'


def _openmdao_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_turbine_fem():
    turbine_dir = _openmdao_dir()
    turbine_dir_str = str(turbine_dir)
    if turbine_dir_str not in sys.path:
        sys.path.insert(0, turbine_dir_str)

    import turbineFEM

    return turbineFEM


def _geometry_filename() -> str:
    return str(_openmdao_dir() / "turbine_blade.STL")

class turbineHeatTransfer(om.ExplicitComponent):
    """Turbine heat-transfer discipline backed by the compiled MATLAB FEM model."""

    def setup(self):
        """Declare heat-transfer inputs, output, and MATLAB runtime handles."""

        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 

        # Coupling output
        self.add_output('Tbulk', val=1000.)

        # MATLAB FEM setup
        import matlab
        turbineFEM = _load_turbine_fem()
        self._double = matlab.double
        self._solveFEM = turbineFEM.initialize()
        self._geometry_filename = _geometry_filename()

    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate bulk temperature from turbine design variables."""

        Tc1, Tc2, Tc3 = inputs['x'][0], inputs['x'][1], inputs['x'][2]
        K = inputs['x'][3]
        hle, hte = inputs['x'][4], inputs['x'][5]

        x_in = self._double(np.array([hle, hte, K, Tc1, Tc2, Tc3], dtype='d'), size=(1,6))

        outputs['Tbulk'] = self._solveFEM.turbineFEM(self._geometry_filename, x_in)

class turbineLifetime(om.ExplicitComponent):
    """Turbine lifetime discipline without feedback coupling."""

    def setup(self):
        """Declare lifetime inputs and time-to-failure output."""

        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 

        # Coupling parameter
        self.add_input('Tbulk', val=1000.)

        # Coupling output
        self.add_output('tfail', val=1000.)

    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate time to failure from Larson-Miller parameter and temperature."""

        Plm = inputs['x'][6]
        Tbulk = inputs['Tbulk']

        outputs['tfail'] = np.exp(Plm/Tbulk - 20)

class turbineLifetime_modified(om.ExplicitComponent):
    """Turbine lifetime discipline with performance feedback."""

    def setup(self):
        """Declare feedback lifetime inputs and time-to-failure output."""

        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 

        # Coupling parameter
        self.add_input('Tbulk', val=1000.)
        self.add_input('Peng', val=5.0e6)

        # Coupling output
        self.add_output('tfail', val=1000.)

    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate feedback-modified time to failure."""

        Plm = inputs['x'][6]
        Tbulk = inputs['Tbulk']
        Peng = inputs['Peng']

        outputs['tfail'] = np.exp(Plm/Tbulk - 20 + 2*(Peng/1e7)**2)

class turbinePerformance(om.ExplicitComponent):
    """Turbine engine-performance discipline without feedback terms."""

    def setup(self):
        """Declare performance inputs and engine-power output."""

        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 

        # Coupling output
        self.add_output('Peng', val=5.0e6)

    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate engine power from mass flow, gas temperature, and factor."""

        mdot, Tg, Fperf = inputs['x'][7], inputs['x'][8], inputs['x'][9]

        # Constants
        mdot0 = 30
        T0 = 300
        N = 90
        Cp = 1003.5

        outputs['Peng'] = Fperf*(mdot0-N*mdot)*Cp*T0*(1+Tg/T0-2*np.sqrt(Tg/T0))

class turbinePerformance_modified(om.ExplicitComponent):
    """Turbine performance discipline with lifetime/economic feedback."""

    def setup(self):
        """Declare feedback performance inputs and engine-power output."""

        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 
        self.add_input('recon', val=1.0e4)
        self.add_input('tfail', val=100)

        # Coupling output
        self.add_output('Peng', val=5.0e6)

    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate feedback-modified engine power."""

        mdot, Tg, Fperf = inputs['x'][7], inputs['x'][8], inputs['x'][9]
        recon = inputs['recon']
        tfail = inputs['tfail']

        # Constants
        mdot0 = 30
        T0 = 300
        N = 90
        Cp = 1003.5

        outputs['Peng'] = Fperf*(mdot0-N*mdot)*Cp*T0*(1+Tg/T0-2*np.sqrt(Tg/T0)) + 100*tfail**2 + 0.0001*recon**2

class turbineEconomics(om.ExplicitComponent):
    """Turbine economics discipline mapping reliability and power to return."""

    def setup(self):
        """Declare economic inputs and return output."""

        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 

        # Coupling parameters
        self.add_input('tfail', val=1000.)
        self.add_input('Peng', val=5.0e6)

        # Coupling output
        self.add_output('recon', val=1000.)

    def setup_partials(self):
        """Declare complex-step finite-difference partial derivatives."""

        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """Evaluate economic return."""

        Fecon = inputs['x'][10]

        tfail = inputs['tfail']
        Peng = inputs['Peng']

        # Constant
        c0 = 0.07

        outputs['recon'] = Fecon*tfail*Peng*(c0/1000)

class turbineGroup(om.Group):
    """OpenMDAO group for the feed-forward turbine model."""

    def setup(self):
        """Assemble feed-forward turbine disciplines."""

        self.add_subsystem('heat', turbineHeatTransfer(), promotes=['*'])
        self.add_subsystem('life', turbineLifetime(), promotes=['*'])
        self.add_subsystem('perf', turbinePerformance(), promotes=['*'])
        self.add_subsystem('econ', turbineEconomics(), promotes=['*'])

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NonlinearBlockGS()

class turbineGroup_feedback(om.Group):
    """OpenMDAO group for the feedback-coupled turbine model."""

    def setup(self):
        """Assemble feedback turbine disciplines and nonlinear solver."""

        self.add_subsystem('heat', turbineHeatTransfer(), promotes=['*'])
        self.add_subsystem('life', turbineLifetime_modified(), promotes=['*'])
        self.add_subsystem('perf', turbinePerformance_modified(), promotes=['*'])
        self.add_subsystem('econ', turbineEconomics(), promotes=['*'])

        self.linear_solver = om.DirectSolver()
        nlbgs=self.nonlinear_solver = om.NonlinearBlockGS()
        nlbgs.options['maxiter'] = 100


class Turbine(MDA):
    """Four-discipline turbine model with heat, life, performance, and economics."""

    group_type = "feedforward"

    def __init__(self):
        self._x = zeros(1, 11)
        self._Tbulk = zeros(1)
        self._tfail = zeros(1)
        self._Peng = zeros(1)
        self._recon = zeros(1)
        self._bounds = tensor(
            [
                [590, 610],
                [640, 660],
                [690, 710],
                [29, 31],
                [1975, 2025],
                [975, 1025],
                [2.45e4, 2.55e4],
                [0.108, 0.132],
                [1225, 1275],
                [0.85, 0.95],
                [0.9, 1.1],
                [1000, 1200],
                [0.01, 100],
                [4.8e6, 6.6e6],
                [0, 1.0e5],
            ]
        ).transpose(0, 1)
        self._dim = 15
        self._input_dim = 11
        self._coupling_dim = 4
        self._tasks = [0, 1, 2, 3]
        self._residual_mode = "all"

        import matlab
        turbineFEM = _load_turbine_fem()

        self._double = matlab.double
        self._solveFEM = turbineFEM.initialize()
        self._geometry_filename = _geometry_filename()

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
        """Heat-transfer residual for bulk blade temperature."""

        Tc1, Tc2, Tc3 = self._x[..., 0], self._x[..., 1], self._x[..., 2]
        K = self._x[..., 3]
        hle, hte = self._x[..., 4], self._x[..., 5]
        x_in = self._double(to_numpy(torch.column_stack([hle, hte, K, Tc1, Tc2, Tc3])), size=(len(self._x), 6))
        fem_value = tensor(self._solveFEM.turbineFEM(self._geometry_filename, x_in)).squeeze()
        return self._Tbulk - fem_value

    @property
    def r2(self) -> torch.Tensor:
        """Lifetime residual on a logarithmic time-to-failure scale."""

        Plm = self._x[..., 6]
        return torch.log(self._tfail) - (Plm / self._Tbulk - 20.0)

    @property
    def r3(self) -> torch.Tensor:
        """Engine performance residual."""

        mdot, Tg, Fperf = self._x[..., 7], self._x[..., 8], self._x[..., 9]
        mdot0 = 30.0
        T0 = 300.0
        N = 90.0
        Cp = 1003.5
        return self._Peng - Fperf * (mdot0 - N * mdot) * Cp * T0 * (1.0 + Tg / T0 - 2.0 * torch.sqrt(Tg / T0))

    @property
    def r4(self) -> torch.Tensor:
        """Economic return residual."""

        Fecon = self._x[..., 10]
        c0 = 0.07
        return self._recon - Fecon * self._tfail * self._Peng * (c0 / 1000.0)

    @property
    def res(self) -> torch.Tensor:
        if self._residual_mode == "all":
            return torch.column_stack([self.r1, self.r2, self.r3, self.r4])
        if self._residual_mode == "diagonal":
            return self._diagonal_residuals()
        raise ValueError(f"Unknown residual mode: {self._residual_mode}")

    def set_res_mode(self, mode: str) -> None:
        """Select whether ``res`` returns all residuals or diagonal residuals only."""

        if mode not in {"all", "diagonal"}:
            raise ValueError("mode must be 'all' or 'diagonal'.")
        self._residual_mode = mode

    def _diagonal_residuals(self) -> torch.Tensor:
        """Evaluate only one residual for each candidate row."""

        x_current = torch.column_stack((self._x, self._Tbulk, self._tfail, self._Peng, self._recon))
        if len(x_current) != self.coupling_dim:
            raise ValueError("Diagonal residual mode requires one row per residual task.")
        residual_getters = [type(self).r1.fget, type(self).r2.fget, type(self).r3.fget, type(self).r4.fget]
        outputs = []
        for row, residual_getter in zip(x_current, residual_getters):
            self.set_vars(row.unsqueeze(0))
            outputs.append(residual_getter(self).squeeze())
        self.set_vars(x_current)
        return torch.diag(torch.stack(outputs))

    def set_vars(self, x: torch.Tensor) -> None:
        x = as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected {self.dim} variables, got {x.shape[1]}.")
        self._x = x[:, : self.input_dim]
        self._Tbulk = x[:, -4]
        self._tfail = x[:, -3]
        self._Peng = x[:, -2]
        self._recon = x[:, -1]

    def _run_OpenMDAO(self, x_input: torch.Tensor):
        """Run the turbine OpenMDAO group for a fixed external input vector."""

        group = turbineGroup_feedback() if self.group_type == "feedback" else turbineGroup()
        prob = om.Problem(group)
        prob.setup()
        prob.set_val("x", to_numpy(as_tensor(x_input)))
        prob.run_model()
        return prob

    def from_OpenMDAO(self, x_input: torch.Tensor) -> torch.Tensor:
        prob = self._run_OpenMDAO(x_input)
        return tensor(
            [
                prob.get_val("Tbulk").item(),
                prob.get_val("tfail").item(),
                prob.get_val("Peng").item(),
                prob.get_val("recon").item(),
            ]
        )

    def __del__(self):
        solver = getattr(self, "_solveFEM", None)
        if solver is not None:
            solver.terminate()


class TurbineFeedback(Turbine):
    """Turbine variant with feedback from performance/economics to lifetime."""

    group_type = "feedback"

    @property
    def r2(self) -> torch.Tensor:
        Plm = self._x[..., 6]
        return torch.log(self._tfail) - ((Plm / self._Tbulk - 20.0) + 2.0 * (self._Peng / 1e7) ** 2)

    @property
    def r3(self) -> torch.Tensor:
        mdot, Tg, Fperf = self._x[..., 7], self._x[..., 8], self._x[..., 9]
        mdot0 = 30.0
        T0 = 300.0
        N = 90.0
        Cp = 1003.5
        return self._Peng - (
            Fperf * (mdot0 - N * mdot) * Cp * T0 * (1.0 + Tg / T0 - 2.0 * torch.sqrt(Tg / T0))
            + 100.0 * self._tfail**2
            + 0.0001 * self._recon**2
        )


Turbine_feedback = TurbineFeedback
