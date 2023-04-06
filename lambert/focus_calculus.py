from optimization import levenbergMarquardt 

from numpy import arctan2
from numpy import cos, sin
from numpy import sqrt
from numpy import zeros

from numpy.linalg import norm

from numpy.random import rand



def get_foci_by_a( a, r_chs_0_ORP, r_trg_t_ORP ):

    _r = norm( r_chs_0_ORP - r_trg_t_ORP )
    r1 = 2*a - norm( r_chs_0_ORP )
    r2 = 2*a - norm( r_trg_t_ORP )

    args = {
        "_r": _r,
        "r1": r1,
        "r2": r2
    }

    f = zeros(2)
    J = zeros((2,2))

    func = FA_func( args, f )
    grad = FA_jacb( args, J )

    x = levenbergMarquardt( func, grad, rand(2) )

    x = r1 * sin( x[0] )

    D = zeros(3)
    D[0] = r_trg_t_ORP[0] - r_chs_0_ORP[0]
    D[1] = r_trg_t_ORP[1] - r_chs_0_ORP[1]
    D[:] = D[:] / norm( D )
    
    O = r_chs_0_ORP + D * sqrt( r1**2 - x**2 )

    D[0] = r_chs_0_ORP[1] - r_trg_t_ORP[1]
    D[1] = r_trg_t_ORP[0] - r_chs_0_ORP[0]

    D[:] = D[:] / norm( D )

    F1 = O + D * x
    F2 = O - D * x

    return F1, F2


def get_elem_by_foci( F, O_orp ):
    ## orbital element
    a = O_orp['a']
    o = O_orp['o']
    i = O_orp['i']
    ## eccentricity
    ae = norm( F )
    e  = ae / ( 2 * a )
    ## eccentricity vector
    w = arctan2( -F[1], -F[0] )

    O_orp['e'] = e
    O_orp['w'] = w


def FA_func( args, out ):

    _r = args["_r"]
    r1 = args["r1"]
    r2 = args["r2"]

    def _func( x ):

        a, b = x

        out[0] = r1 * cos( a ) + r2 * cos( b ) - _r
        out[1] = r1 * sin( a ) - r2 * sin( b )

        return out

    return _func


def FA_jacb( args, out ):

    _r = args["_r"]
    r1 = args["r1"]
    r2 = args["r2"]

    def _jacb( x ):

        a, b = x

        out[0,0] = r1 * sin( a ) * ( -1 )
        out[1,0] = r1 * cos( a ) 
        out[0,1] = r2 * sin( b ) * ( -1 )
        out[1,1] = r2 * cos( b ) * ( -1 )

        return out
    
    return _jacb