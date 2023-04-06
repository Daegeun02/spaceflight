from coordinate import ECI2PQW

from numpy import cos, sin
from numpy import arctan2
from numpy import sqrt
from numpy import zeros



def impulse_ctrl( r_xxx_x_ECI, v_xxx_x_ECI, O_orp, mu ):
    ## orbital element
    a = O_orp['a']
    e = O_orp['e']
    o = O_orp['o']
    i = O_orp['i']
    w = O_orp['w']

    p = a * ( 1 - e**2 )
    C = sqrt( mu / p )

    R = ECI2PQW( o, i, w )

    r_xxx_x_PQW = R @ r_xxx_x_ECI

    f = arctan2( r_xxx_x_PQW[1], r_xxx_x_PQW[0] )

    v_xxx_x_PQW = zeros(3)
    v_xxx_x_PQW[0] = ( -C ) * sin( f )
    v_xxx_x_PQW[1] = C * ( e + cos( f ) )

    Dv_    = zeros(3)
    Dv_[:] = ( R.T @ v_xxx_x_PQW ) - v_xxx_x_ECI

    return Dv_