import numpy as np
import openmdao.api as om
import os

os.environ['OPENMDAO_REPORTS'] = 'none'
os.environ['OPENMDAO_USE_MPI'] = 'false'

class satelliteDis1(om.ExplicitComponent):
    def setup(self):
        # Global Design Variable
        self.add_input('x', val=np.ones(5))
        
        # Coupling parameter
        self.add_input('u21', val=9.)

        # Coupling output
        self.add_output('u12', val=9.)

    def setup_partials(self):
        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        x1 = inputs['x'][0]
        x2 = inputs['x'][1]
        x3 = inputs['x'][2]
        u21 = inputs['u21']

        if u21.real < 0.0:
            u21 *= -1

        outputs['u12'] = x1**2 + 2*x2 - x3 + 2*(u21**0.5)

class satelliteDis2(om.ExplicitComponent):
    def setup(self):
        # Global Design Variable
        self.add_input('x', val=np.ones(5))
        
        # Coupling parameter
        self.add_input('u12', val=9.)

        # Coupling output
        self.add_output('u21', val=9.)
        
    def setup_partials(self):
        # Finite difference all partials
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        x1 = inputs['x'][0]
        x4 = inputs['x'][3]
        x5 = inputs['x'][4]
        u12 = inputs['u12']

        outputs['u21'] = x1*x4 + x4**2 + x5 + u12
        

class satelliteGroup(om.Group):
    def setup(self):
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