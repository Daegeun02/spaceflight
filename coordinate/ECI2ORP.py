from numpy import cos, sin
from numpy import zeros



def ECI2ORP( o, i ):

    R = zeros((3,3))

    co, so = cos( o ), sin( o )
    ci, si = cos( i ), sin( i )

    R[0,0] = co
    R[1,0] = (-1) * so * ci
    R[2,0] = so * si
    R[0,1] = so
    R[1,1] = co * ci
    R[2,1] = (-1) * co * si
    R[0,2] = 0
    R[1,2] = si
    R[2,2] = ci

    return R