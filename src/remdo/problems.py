"""Problem definitions for coupled multidisciplinary residual systems.

The classes in this module expose a common residual-evaluation interface used
by REMDO's GP training and active-learning routines.  A problem stores bounds
for the full variable vector ``[external inputs, coupling variables]`` and
returns one residual column per coupling equation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from .config import as_tensor, tensor, to_numpy, zeros
from .openmdao_loader import load_openmdao_symbol


class MDA(ABC):
    """Abstract base class for multidisciplinary-analysis residual problems."""

    @property
    @abstractmethod
    def bounds(self) -> torch.Tensor:
        """``2 x dim`` lower/upper bounds for all problem variables."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Total number of external plus coupling variables."""

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Number of external/input variables."""

    @property
    @abstractmethod
    def coupling_dim(self) -> int:
        """Number of coupling variables and residual tasks."""

    @property
    @abstractmethod
    def tasks(self) -> list[int]:
        """Task ids corresponding to residual columns."""

    @property
    @abstractmethod
    def res(self) -> torch.Tensor:
        """Residual matrix for the most recent variables passed to ``set_vars``."""

    @abstractmethod
    def set_vars(self, x: torch.Tensor) -> None:
        """Set the full problem variable matrix used by residual properties."""

    def set_bounds(self, bounds: torch.Tensor) -> None:
        """Set full-variable bounds after validating the expected shape."""

        bounds = as_tensor(bounds)
        if bounds.shape != (2, self.dim):
            raise ValueError(f"Expected bounds shape {(2, self.dim)}, got {tuple(bounds.shape)}.")
        self._bounds = bounds

    @abstractmethod
    def from_OpenMDAO(self, x_input: torch.Tensor) -> torch.Tensor:
        """Solve the coupled OpenMDAO model for a fixed external input."""


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

        import numpy as np
        import openmdao.api as om

        satelliteGroup = load_openmdao_symbol("satellite_openmdao.py", "satelliteGroup")
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

        import openmdao.api as om

        aerostructuresGroup = load_openmdao_symbol("aerostructures_openmdao.py", "aerostructuresGroup")
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

        from . import turbineFEM
        import matlab

        self._double = matlab.double
        self._solveFEM = turbineFEM.initialize()
        self._geometry_filename = str(Path(__file__).with_name("turbine_blade.STL"))

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

        import openmdao.api as om

        turbineGroup = load_openmdao_symbol("turbine_openmdao.py", "turbineGroup")
        turbineGroup_feedback = load_openmdao_symbol("turbine_openmdao.py", "turbineGroup_feedback")
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


# Backward-compatible class names used by older notebooks.
Satellite_direct = SatelliteDirect
Satellite_modified = SatelliteModified
Satellite_modified_3dis = SatelliteModified3Dis
Turbine_feedback = TurbineFeedback
