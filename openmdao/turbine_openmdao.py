import numpy as np
import openmdao.api as om
import os
from pathlib import Path

import remdo

os.environ['OPENMDAO_REPORTS'] = 'none'
os.environ['OPENMDAO_USE_MPI'] = 'false'

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
        from remdo import turbineFEM
        import matlab
        self._double = matlab.double
        self._solveFEM = turbineFEM.initialize()
        package_dir = Path(remdo.__file__).resolve().parent
        self._geometry_filename = str(package_dir / "turbine_blade.STL")

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
