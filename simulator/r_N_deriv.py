## derivatives of 'r' and 'nu'
from runge_kutta import _RK4, _A_RK4

from numpy import sqrt
from numpy import sin, cos
from numpy import arctan2



def deriv_r( args ):

    p = args["p"]
    m = args["mu"]
    e = args["e"]

    def dr_dt( t, r, args ):

        N = args["N"]

        return sqrt( m / p ) * e * sin( N )

    return dr_dt


def deriv_N( args ):

    p = args["p"]
    m = args["mu"]
    e = args["e"]

    p3 = p ** 3

    def dN_dt( t, N, args ):

        return sqrt( m / p3 ) * ( ( 1 + e * cos( N ) ) ** 2 )

    return dN_dt


def _r_N_setup( satellite, geometric ):

    print("initializing")

    a = satellite.a
    e = satellite.e

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

    return dN_dt, dr_dt


def _r_N_run( dN_dt, dr_dt, R_at_pqw, position, dt ):

    R_at_pqw["N"] = _RK4( dN_dt, -1, R_at_pqw["N"], dt, args=R_at_pqw )
    R_at_pqw["r"] = _RK4( dr_dt, -1, R_at_pqw["r"], dt, args=R_at_pqw )

    cN = cos(R_at_pqw["N"])
    sN = sin(R_at_pqw["N"])
    
    R = R_at_pqw["r"]

    position.append([R*cN, R*sN])