from numpy import sqrt
from numpy import arctan2
from numpy import cos, sin



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

        return out
    
    return dr_dt