from threading import Thread

from .derivatives import deriv_N, deriv_r

from .runge_kutta import _RK4, _A_RK4

from two_body_problem import TwoBodyOrbit
from two_body_problem import step_1, step_2, step_3, step_4

from numpy import cos, sin, pi
from numpy import sqrt

from time import sleep



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

        self.position = []

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

        position = self.position

        ## geometric model
        mu  = geometric.mu

        satellite.period = 2 * pi * sqrt( ( a ** 3 ) / mu )

        print(f"period is {satellite.period}")

        dt = self.dt

        p = a * ( 1 - ( e ** 2 ) )

        args = {
            "e" : e,
            "M" : -1,
            "n" : sqrt( mu / ( a ** 3 ) ),
            "E" : -1,
            "eC": sqrt( ( 1 + e ) / ( 1 - e ) ),
            "N" : -1,
            "p" : p,
            "pC": sqrt( mu / p ),
            "T" : T
        }

        print("initialize finished...")
        print("start simulation")

        while self.simulating:

            step_1( args, self.t )

            step_2( args )

            step_4( args, position )

            self.t += dt
        
        print(self.t)


    # def run(self):

    #     print("initializing")

    #     satellite = self.satellite
    #     geometric = self.geometric

    #     position = self.position

    #     dt = self.dt

    #     a = satellite.a
    #     e = satellite.e

    #     ## state
    #     R_at_pqw = satellite.R_at_pqw

    #     args = { 
    #         "p" :  a * ( 1 - ( e ** 2 ) ),
    #         "mu": geometric.mu,
    #         "N" : satellite.N,
    #         "e" : satellite.e
    #     }

    #     dN_dt = deriv_N( args )
    #     dr_dt = deriv_r( args )

    #     print("initialize finished...")
    #     print("start simulation")

    #     t = 0

    #     while self.simulating:

    #         R_at_pqw["N"] = _RK4( dN_dt, -1, R_at_pqw["N"], dt, args=R_at_pqw )
    #         R_at_pqw["r"] = _RK4( dr_dt, -1, R_at_pqw["r"], dt, args=R_at_pqw )

    #         # R_at_pqw["N"] = _A_RK4( dN_dt, -1, R_at_pqw["N"], dt, args=R_at_pqw )
    #         # R_at_pqw["r"] = _A_RK4( dr_dt, -1, R_at_pqw["r"], dt, args=R_at_pqw )

    #         cN = cos(R_at_pqw["N"])
    #         sN = sin(R_at_pqw["N"])
            
    #         R = R_at_pqw["r"]

    #         position.append([R*cN, R*sN])

    #         t += dt

    #     self.t = t

    #     print(self.t)


    def stop(self):

        self.simulating = False

        print("simulation finished")

        self.join()