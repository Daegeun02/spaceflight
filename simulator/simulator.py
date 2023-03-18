from .derivatives import deriv_N, deriv_r

from .runge_kutta import _RK4



class Simulator:
    '''
    Simulates satellite's movement.

    With Runge-Kutta 4th order method,
    numerically integrate satellite's state.

    State
    r : distance from the focus of orbit
    nu: True anomaly
    '''


    def __init__(self, satellite, geometric):

        self.satellite = satellite
        self.geometric = geometric

        self.simulating = True


    def start(self, dt):

        satellite = self.satellite
        geometric = self.geometric

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

        while self.simulating:

            _RK4( dN_dt, -1, R_at_pqw["N"], dt )
            _RK4( dr_dt, -1, R_at_pqw["r"], dt )