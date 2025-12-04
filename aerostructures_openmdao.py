import numpy as np
import openmdao.api as om
import os

os.environ['OPENMDAO_REPORTS'] = 'none'


class aerodynamicsDis(om.ExplicitComponent):
    def setup(self):
        # Global Design Variable
        self.add_input('B', val=0.)

        # Coupling parameter
        self.add_input('phi', val=0.)

        # Coupling output
        self.add_output('L', val=0.)

    def setup_partials(self):
        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        q = 1 # N/cm2
        C = 10 # cm
        psi = 0.05 # rad
        r = 0.9425
        theta0 = 0.26 # rad

        B = inputs['B']
        phi = inputs['phi']

        outputs['L'] = 1/1000 * q*B*C * ((2*np.pi*(phi+psi)) + r*(1-np.cos(np.pi/2*(phi+psi)/theta0)))

class structuresDis(om.ExplicitComponent):
    def setup(self):
        # Coupling parameter
        self.add_input('L', val=0.)

        # Coupling output
        self.add_output('phi', val=0.)

    def setup_partials(self):
        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        C = 10 # cm
        p = 0.1111
        k1 = 4000 # N/cm
        k2 = 2000 # N/cm
        z1 = 0.2
        z2 = 0.7

        L = inputs['L']

        outputs['phi'] = np.remainder(( 1000*L/(k1*(1+p)) - (1000*L*p)/(k2*(1+p)) ) * ( 1/(C*(z2-z1)) ), 2*np.pi)
        
class aerostructuresGroup(om.Group):
    def setup(self):
        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('aero', aerodynamicsDis(), promotes_inputs=['B', 'phi'],
                            promotes_outputs=['L'])
        cycle.add_subsystem('strux', structuresDis(), promotes_inputs=['L'],
                            promotes_outputs=['phi'])

        cycle.linear_solver = om.DirectSolver(rhs_checking=False)
        nlbgs = cycle.nonlinear_solver = om.NonlinearBlockGS()
        nlbgs.options['maxiter'] = 1000
        nlbgs.options['iprint'] = 0
        