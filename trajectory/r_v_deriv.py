## calculate derivative of position and velocity on ECI coordinate system
from numpy import sqrt
from numpy import arctan2
from numpy import cos, sin

from numpy.linalg import norm



def deriv_x( args, out ):

    m = args["mu"]

    def dx_dt( t, x, args ):

        position = x[:3]
        velocity = x[3:]

        r_cube = norm( position ) ** 3

        out[:3] = velocity
        out[3:] = (-1) * ( m / r_cube ) * position

        return out

    return dx_dt


def deriv_r( args, out ):
    
    m = args["mu"]
    p = args["p"]
    e = args["e"]

    c = sqrt( m / p )

    def dr_dt( t, r, args ):

        rp, rq, rw = r

        N = arctan2( rq, rp )

        out[0] = c * sin( N ) * (-1)
        out[1] = c * ( e + cos( N ) )
    
    return dr_dt


def deriv_v( args, out ):

    m = args["mu"]

    def dv_dt( t, v, args ):

        position = args["position"]

        r_cube = norm( position ) ** 3

        out[:] = (-1) * ( m / r_cube ) * position[:]

    return dv_dt