from numpy import cos, sin
from numpy import zeros



def ECI2PQW( o, i, w ):

    R = zeros((3,3))
    
    co, so = cos( o ), sin( o )
    ci, si = cos( i ), sin( i )
    cw, sw = cos( w ), sin( w )

    R[0,0] = cw * co - sw * ci * so
    R[1,0] = (-1) * (sw * co + cw * ci * so)
    R[2,0] = si * so
    R[0,1] = cw * so + sw * ci * co
    R[1,1] = cw * ci * co - sw * so
    R[2,1] = (-1) * si * co
    R[0,2] = sw * si
    R[1,2] = cw * si
    R[2,2] = ci

    return R