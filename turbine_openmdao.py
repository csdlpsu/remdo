import numpy as np
import openmdao.api as om
import os

os.environ['OPENMDAO_REPORTS'] = 'none'
os.environ['OPENMDAO_USE_MPI'] = 'false'

class turbineHeatTransfer(om.ExplicitComponent):
    def setup(self):
        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 

        # Coupling output
        self.add_output('Tbulk', val=1000.)

        # MATLAB FEM setup
        import turbineFEM
        import matlab
        self._double = matlab.double
        self._solveFEM = turbineFEM.initialize()
        self._geometry_filename = "turbine_blade.STL"

    def setup_partials(self):
        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Tc1, Tc2, Tc3 = inputs['x'][0], inputs['x'][1], inputs['x'][2]
        K = inputs['x'][3]
        hle, hte = inputs['x'][4], inputs['x'][5]

        x_in = self._double(np.array([hle, hte, K, Tc1, Tc2, Tc3], dtype='d'), size=(1,6))

        outputs['Tbulk'] = self._solveFEM.turbineFEM(self._geometry_filename, x_in)

class turbineLifetime(om.ExplicitComponent):
    def setup(self):
        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 

        # Coupling parameter
        self.add_input('Tbulk', val=1000.)

        # Coupling output
        self.add_output('tfail', val=1000.)

    def setup_partials(self):
        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Plm = inputs['x'][6]
        Tbulk = inputs['Tbulk']

        outputs['tfail'] = np.exp(Plm/Tbulk - 20)

class turbinePerformance(om.ExplicitComponent):
    def setup(self):
        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 

        # Coupling output
        self.add_output('Peng', val=5.0e6)

    def setup_partials(self):
        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        mdot, Tg, Fperf = inputs['x'][7], inputs['x'][8], inputs['x'][9]

        # Constants
        mdot0 = 30
        T0 = 300
        N = 90
        Cp = 1003.5

        outputs['Peng'] = Fperf*(mdot0-N*mdot)*Cp*T0*(1+Tg/T0-2*np.sqrt(Tg/T0))

class turbineEconomics(om.ExplicitComponent):
    def setup(self):
        # Global design variable
        # [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
        self.add_input('x', val=np.ones(11)) 

        # Coupling parameters
        self.add_input('tfail', val=1000.)
        self.add_input('Peng', val=5.0e6)

        # Coupling output
        self.add_output('recon', val=1000.)

    def setup_partials(self):
        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Fecon = inputs['x'][10]

        tfail = inputs['tfail']
        Peng = inputs['Peng']

        # Constant
        c0 = 0.07

        outputs['recon'] = Fecon*tfail*Peng*(c0/1000)

class turbineGroup(om.Group):
    def setup(self):
        self.add_subsystem('heat', turbineHeatTransfer(), promotes=['*'])
        self.add_subsystem('life', turbineLifetime(), promotes=['*'])
        self.add_subsystem('perf', turbinePerformance(), promotes=['*'])
        self.add_subsystem('econ', turbineEconomics(), promotes=['*'])