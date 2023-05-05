from geometry import MU

from coordinate import ECI2PQW

from .r_v_deriv import *

from .runge_kutta import _RK4

from numpy import pi
from numpy import zeros, ndarray



def elliptic_orbit( OrbitalElement, r_xxx_x_ECI=None, v_xxx_x_ECI=None, dt=1, rev=1, impulse=None ):

    a = OrbitalElement["a"]
    try:
        o = OrbitalElement["o"]
        i = OrbitalElement["i"]
        w = OrbitalElement["w"]
        R = ECI2PQW( o, i, w )
    except KeyError:
        R = OrbitalElement["R"]
    e = OrbitalElement["e"]
    # T = OrbitalElement["T"]

    ## period for a revolution
    period = 2 * pi * sqrt( ( a**3 ) / MU )
    ## number of time steps for a revolution
    N = int( rev * period / dt )

    ## trajectory
    pos = zeros((N,3))
    vel = zeros((N,3))
    ## state of satellite
    x    = zeros(6)
    xdot = zeros(6)

    ## config parameters
    p    = a * ( 1 - ( e**2 ) )
    args = {
        'mu': MU,
        'p' : p,
        'e' : e
    }

    ## gradient function
    dxdt = deriv_x( args, xdot )

    ## simulate
    t = 0

    ## initialize state
    if ( ( type( r_xxx_x_ECI ) == ndarray ) and ( type( v_xxx_x_ECI ) == ndarray ) ):
        x[0:3] = r_xxx_x_ECI
        x[3:6] = v_xxx_x_ECI
    elif ( type( r_xxx_x_ECI ) == ndarray ):
        x[0:3] = r_xxx_x_ECI
        r_xxx_x_PQW = R @ r_xxx_x_ECI
        f = arctan2( r_xxx_x_PQW[1], r_xxx_x_PQW[0] )
        x[3] = sqrt( MU / p ) * ( -sin( f ) )
        x[4] = sqrt( MU / p ) * ( e + cos( f ) )
        x[3:6] = R.T @ x[3:6]
    else:
        x[0] = a * ( 1 - e )
        x[4] = sqrt( MU / p ) * ( e + 1 )

        x[0:3] = R.T @ x[0:3]
        x[3:6] = R.T @ x[3:6]

    pos[0,:] = x[0:3]
    vel[0,:] = x[3:6]

    if ( impulse != None ):
        thr_t = list( impulse.keys() )
    else:
        thr_t = []

    for k in range(N-1):
        _RK4( dxdt, t, x, dt, args=args )

        pos[k+1,:] = x[0:3]
        vel[k+1,:] = x[3:6]

        if ( t in thr_t ):
            x[3:6] += impulse[t]

        t += dt

    return pos, vel