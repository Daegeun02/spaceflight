from numpy import cos, sin
from numpy import arctan2
from numpy import sqrt
from numpy import zeros

from numpy.linalg import norm



def impulse_ctrl( F, r_xxx_x_ORP, v_xxx_x_ORP, O_orp, mu ):
    ## orbital element
    a = O_orp['a']
    o = O_orp['o']
    i = O_orp['i']
    ## angular position
    theta = arctan2( r_xxx_x_ORP[1], r_xxx_x_ORP[0] )
    ## find eccentricity and argument perigee
    ae = norm( F )
    e  = ae / a
    w  = arctan2( F[1], F[0] )
    ## true anomaly
    N = theta - w
    p = a * ( 1 - e**2 )
    C = sqrt( mu / p )
    ## velocity for enter the transfer orbit
    v_xxx_x_TRF = zeros(3)
    v_xxx_x_TRF[0] = ( -C ) * sin( N )
    v_xxx_x_TRF[1] = C * ( e + cos( N ) )
    ## impulse control
    Dv_    = zeros(3)
    Dv_[:] = v_xxx_x_TRF - v_xxx_x_ORP
    ## store orbital element
    O_orp['e'] = e
    O_orp['w'] = w

    return Dv_