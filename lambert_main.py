from trajectory import elliptic_orbit

from lambert import Build_LP_solver

from two_body_problem import estimate

from transfer import UF_FG_S

from geometry import MU
from geometry import EARTHRADS as Re

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import pandas as pd

from numpy import deg2rad
from numpy import array
from numpy import linspace
from numpy import cos, sin

from itertools import product



t_tof = 17500
t_chs = 0.0
t_trg = 0.0

O_chs = {
    "a": 10000,
    "o": deg2rad( 30 ),
    "i": deg2rad( 0 ),
    "w": deg2rad( 0 ),
    "e": 0.1,
    "T": 0
}

O_trg = {
    "a": 15000,
    "o": deg2rad( 0 ),
    "i": deg2rad( 30 ),
    "w": deg2rad( 30 ),
    "e": 0.5,
    "T": 0
}




if __name__ == "__main__":

    ## estimate chaser's position and velocity
    r_chs_0_ECI, v_chs_0_ECI = estimate( O_chs, MU, t_chs )
    ## estimate target's initial position and velocity
    r_trg_0_ECI, v_trg_0_ECI = estimate( O_trg, MU, t_trg )

    print( r_chs_0_ECI, v_chs_0_ECI )
    print( r_trg_0_ECI, v_trg_0_ECI )

    # _chs_0_ECI = array([5000e3, 0, 5042e3, 0, 7492.18085, 0])
    # _trg_0_ECI = array([2134089.51, 4094615.131, 5655178.001, -4451.749557, -3952.959997, 4930.183112])
    # _chs_t_ECI = array([4.36602877e6, 2.43675866e6, 5.04200000e6, 3.17506572e2, 1.77206090e2, 0.00000000e0])
    # _trg_t_ECI = array([4.36602877e6, 2.43675866e6, 5.04200000e6, 3.17506572e2, 1.77206090e2, 0.00000000e0])

    # t_tof = 7000

    # r_chs_0_ECI = _chs_0_ECI[0:3] / 1000
    # v_chs_0_ECI = _chs_0_ECI[3:6] / 1000

    # r_trg_t_ECI = _trg_t_ECI[0:3] / 1000
    # v_trg_t_ECI = _trg_t_ECI[3:6] / 1000

    ## solve Lambert Problem
    _solver = Build_LP_solver( 
        r_chs_0_ECI, v_chs_0_ECI, r_trg_0_ECI, v_trg_0_ECI, MU, O_chs, O_trg
    )

    O_orp, Dv0, Dv1, F = _solver( t_tof )

    print( O_orp )
    print( Dv0, Dv1 )

    '''
    impulse = {
        0    : Dv0,
        t_tof: Dv1
    }
    ## simulate chaser's original orbit
    pos_chs  , vel_chs   = elliptic_orbit( O_chs, rev=1 )
    ## simulate chaser's transfer orbit
    pos_orp  , vel_orp   = elliptic_orbit( O_orp, r_chs_0_ECI, v_chs_0_ECI, rev=1, impulse=impulse)
    ## simulate chaser's transfer orbit
    pos_orp_o, vel_orp_o = elliptic_orbit( O_orp, r_chs_0_ECI, rev=1 )
    ## simulate target's orbit while doing transfer
    pos_trg  , vel_trg   = elliptic_orbit( O_trg, rev=1 )

    ## plot result
    ## modify legend freely
    fig = plt.figure( figsize=( 8,8 ) )
    ax  = plt.axes( projection='3d' )

    # ax.scatter(
    #     0,
    #     0,
    #     0,
    #     label=' earth ',
    #     s=100
    # )
    draw_earth()

    ax.scatter(
        F[0],
        F[1],
        F[2],
        label=' focus 1 '
    )

    ax.plot3D(
        pos_chs[:,0],
        pos_chs[:,1],
        pos_chs[:,2],
        label=' chaser ',
        color='b',
        alpha=0.3
    )

    ax.plot3D(
        pos_chs[0:t_chs,0],
        pos_chs[0:t_chs,1],
        pos_chs[0:t_chs,2],
        label=' chaser ',
        color='b'
    )

    ax.scatter( 
        r_chs_0_ECI[0],
        r_chs_0_ECI[1],
        r_chs_0_ECI[2],
        label=" chaser's initial point ",
        color='b'
    )

    ax.scatter(
        pos_orp[t_tof,0],
        pos_orp[t_tof,1],
        pos_orp[t_tof,2],
        color='g',
        s=30
    )

    ax.plot3D(
        pos_orp_o[:t_tof,0],
        pos_orp_o[:t_tof,1],
        pos_orp_o[:t_tof,2],
        label=' transfer orbit ',
        color='g',
        alpha=0.3
    )

    ax.scatter(
        pos_orp_o[0,0],
        pos_orp_o[0,1],
        pos_orp_o[0,2],
        label=' transfer ',
        color='g'
    )

    ax.plot3D(
        pos_orp[:t_tof,0],
        pos_orp[:t_tof,1],
        pos_orp[:t_tof,2],
        label=' transfer trajectory ',
        color='g'
    )

    ax.plot3D(
        pos_orp[t_tof:t_tof+5000,0],
        pos_orp[t_tof:t_tof+5000,1],
        pos_orp[t_tof:t_tof+5000,2],
        label=' 5000s after arrive at rendezvous point ',
        color='orange'
    )

    ax.plot3D(
        pos_trg[:,0],
        pos_trg[:,1],
        pos_trg[:,2],
        label=' target orbit ',
        color='r',
        alpha=0.3
    )

    ax.plot3D(
        pos_trg[t_trg:t_trg+t_tof,0],
        pos_trg[t_trg:t_trg+t_tof,1],
        pos_trg[t_trg:t_trg+t_tof,2],
        label=' target trajectory ',
        color='r'
    )

    ax.scatter(
        r_trg_t_ECI[0],
        r_trg_t_ECI[1],
        r_trg_t_ECI[2],
        label=" rendezvous point ",
        color='r'
    )

    ax.axis('equal')
    
    ax.set_xlabel( 'x-inertia' )
    ax.set_ylabel( 'y-inertia' )

    ax.legend()

    plt.show()
    '''