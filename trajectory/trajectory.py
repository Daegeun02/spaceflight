from geometry import MU

from coordinate import ECI2PQW

from .r_v_deriv import *

from .runge_kutta import _RK4

from numpy import pi
from numpy import zeros



def elliptic_orbit( OrbitalElement, dt=1, rev=1, impulse=None ):

    a = OrbitalElement["a"]
    o = OrbitalElement["o"]
    i = OrbitalElement["i"]
    w = OrbitalElement["w"]
    e = OrbitalElement["e"]
    T = OrbitalElement["T"]

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
        'e' : e,
        'control': zeros(3)
    }

    ## gradient function
    dxdt = deriv_x( args, xdot )

    ## simulate
    t = 0
    ## represent in ECI coordinate system
    R = ECI2PQW( o, i, w )

    ## initialize state
    x[0] = a * ( 1 - e )
    x[4] = sqrt( MU / p ) * ( e + 1 )

    x[0:3] = R.T @ x[0:3]
    x[3:6] = R.T @ x[3:6]

    pos[0,:] = x[0:3]
    vel[0,:] = x[3:6]

    if ( impulse != None ):
        thr_t = [impulse.keys()]

    for k in range(N-1):
        _RK4( dxdt, t, x, dt, args=args )

        pos[k+1,:] = x[0:3]
        vel[k+1,:] = x[3:6]

        if ( t in thr_t ):
            args["control"] = impulse[t]

        t += dt

    return pos, vel