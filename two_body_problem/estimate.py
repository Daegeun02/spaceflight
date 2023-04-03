from coordinate import ECI2PQW

from .steps import *

from numpy import sqrt
from numpy import zeros

from numpy.linalg import norm



def estimate( O, mu, t ):

    a = O["a"]
    o = O["o"]
    i = O["i"]
    w = O["w"]
    e = O["e"]
    T = O["T"]

    p = a * ( 1 - ( e**2 ) )

    args = {
        "e" : e,
        "M" : -1,
        "n" : sqrt( mu / ( a ** 3 ) ),
        "E" : -1,
        "eC": sqrt( ( 1 + e ) / ( 1 - e) ),
        "N" : -1,
        "p" : p,
        "pC": sqrt( mu / p ),
        "T" : T
    }

    _func = func( args )
    _grad = grad( args )

    args["func"] = _func
    args["grad"] = _grad

    r_xxx_t_PQW = zeros(3)
    v_xxx_t_PQW = zeros(3)

    step_1( args, t )
    step_2( args )
    step_3( args, r_xxx_t_PQW, v_xxx_t_PQW )

    R = ECI2PQW( o, i, w )

    r_xxx_t_ECI = R.T @ r_xxx_t_PQW
    v_xxx_t_ECI = R.T @ v_xxx_t_PQW

    return r_xxx_t_ECI, v_xxx_t_ECI


def estimate_with_foci( F1, F2, a, mu, t ):

    ae = norm( F2 - F1 )

    e = ae / a

    p = a * ( 1 - ( e**2 ) )

    args = {
        "e" : e,
        "M" : -1,
        "n" : sqrt( mu / ( a ** 3 ) ),
        "E" : -1,
        "eC": sqrt( ( 1 + e ) / ( 1 - e) ),
        "N" : -1,
        "p" : p,
        "pC": sqrt( mu / p ),
        "T" : 0
    }

    _func = func( args )
    _grad = grad( args )

    args['func'] = _func
    args['grad'] = _grad

    r_xxx_t_PQW = zeros(3)
    v_xxx_t_PQW = zeros(3)

    step_1( args, t )
    step_2( args )
    step_3( args, r_xxx_t_PQW, v_xxx_t_PQW )

    return r_xxx_t_PQW, v_xxx_t_PQW