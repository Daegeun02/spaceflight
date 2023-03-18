from threading import Thread

from .derivatives import deriv_N, deriv_r

from .runge_kutta import _RK4, _A_RK4

from numpy import cos, sin



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

        print("initializing")

        satellite = self.satellite
        geometric = self.geometric

        position = self.position

        dt = self.dt

        a = satellite.a
        e = satellite.e

        ## state
        R_at_pqw = satellite.R_at_pqw

        args = { 
            "p" :  a * ( 1 - ( e ** 2 ) ),
            "mu": geometric.mu,
            "N" : satellite.N,
            "e" : satellite.e
        }

        dN_dt = deriv_N( args )
        dr_dt = deriv_r( args )

        print("initialize finished...")
        print("start simulation")

        t = 0

        while self.simulating:

            # R_at_pqw["N"] = _RK4( dN_dt, -1, R_at_pqw["N"], dt, args=R_at_pqw )
            # R_at_pqw["r"] = _RK4( dr_dt, -1, R_at_pqw["r"], dt, args=R_at_pqw )

            R_at_pqw["N"] = _A_RK4( dN_dt, -1, R_at_pqw["N"], dt, args=R_at_pqw )
            R_at_pqw["r"] = _A_RK4( dr_dt, -1, R_at_pqw["r"], dt, args=R_at_pqw )

            cN = cos(R_at_pqw["N"])
            sN = sin(R_at_pqw["N"])
            
            R = R_at_pqw["r"]

            position.append([R*cN, R*sN])

            t += dt

        self.t = t

        print(self.t)


    def stop(self):

        self.simulating = False

        print("simulation finished")

        self.join()