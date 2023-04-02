from .steps import *

from numpy import sqrt
from numpy import zeros



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


def ECI2PQW( o, i, w ):

    R = zeros((3,3))
    
    co, so = cos( o ), sin( o )
    ci, si = cos( i ), sin( i )
    cw, sw = cos( w ), sin( w )

    R[0,0] = cw * co - sw * ci * so
    R[1,0] = cw * so + sw * ci * co
    R[2,0] = sw * si
    R[0,1] = (-1) * (sw * co + cw * ci * so)
    R[1,1] = cw * ci * co - sw * so
    R[2,1] = cw * si
    R[0,2] = si * so
    R[1,2] = (-1) * si * co
    R[2,2] = ci

    return R