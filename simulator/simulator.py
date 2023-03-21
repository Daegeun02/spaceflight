from threading import Thread

from .r_v_deriv import deriv_x

from .runge_kutta import _RK4, _A_RK4

from numpy import pi
from numpy import sqrt
from numpy import array, zeros



class Simulator(Thread):
    '''
    Simulates satellite's movement.

    With Runge-Kutta 4th order method,
    numerically integrate satellite's state.

    State
    r : distance from the focus of orbit
    nu: True anomaly
    '''


    def __init__(self, satellite, geometric, dt):

        super().__init__()

        self.daemon = True

        self.satellite = satellite
        self.geometric = geometric

        self.trajectory = []

        self.dt = dt
        self.t  = 0

        self.simulating = True


    def run(self):

        print('initialize...')
        ## satellite object
        satellite = self.satellite
        geometric = self.geometric
        ## Classical Orbital Elements
        a = satellite.a
        e = satellite.e
        T = satellite.T
        o = satellite.o
        i = satellite.i
        w = satellite.w

        trajectory = self.trajectory

        ## geometric model
        mu  = geometric.mu

        satellite.period = 2 * pi * sqrt( ( a ** 3 ) / mu )

        print(f"period is {satellite.period}")

        dt = self.dt

        p = a * ( 1 - ( e ** 2 ) )

        args = {
            "mu": mu,
            "e" : e,
            "p" : p,
            "control": zeros(3)
        }

        satellite.init_state( args )

        x    = satellite.state
        xdot = zeros(6)


        dx_dt = deriv_x( args, xdot )

        print("initialize finished...")
        print("start simulation")

        while self.simulating:

            _RK4( dx_dt, self.t, x, dt, args=args )

            trajectory.append(
                array( x )
            )

            self.t += dt
        
        print(self.t)


    def stop(self):

        self.simulating = False

        print("simulation finished")

        self.join()