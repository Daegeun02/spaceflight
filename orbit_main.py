from trajectory import elliptic_orbit

from two_body_problem import estimate
from two_body_problem import estimate_with_foci

from transfer import step01, ECI2ORP
from transfer import LambertProblem
from transfer import get_foci_by_a

from geometry import MU

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from numpy import deg2rad
from numpy import arctan2
from numpy import cross

from numpy.linalg import norm



t_tof = 2000

O_chs = {
    "a": 7000,
    "o": deg2rad( 30 ),
    "i": deg2rad( 30 ),
    "w": deg2rad( 30 ),
    "e": 0,
    "T": 0
}

O_trg = {
    "a": 10000,
    "o": deg2rad( 30 ),
    "i": deg2rad( 30 ),
    "w": deg2rad( 30 ),
    "e": 0.5,
    "T": 0
}


def solve_LP( r_chs_0_ECI, r_trg_t_ECI, t_tof, mu ):

    ## direction of angular momentum vector
    H = cross( r_chs_0_ECI, r_trg_t_ECI )
    h = H / norm( H )

    ## define orbital plane
    o = arctan2(         -h[0], h[1] )
    i = arctan2( norm( h[:2] ), h[2] ) * (-1)

    R = ECI2ORP( o, i )

    r_chs_0_ORP = R @ r_chs_0_ECI
    r_trg_t_ORP = R @ r_trg_t_ECI

    theta_0 = arctan2( r_chs_0_ORP[1], r_chs_0_ORP[0] )
    theta_t = arctan2( r_trg_t_ORP[1], r_trg_t_ORP[0] )
    theta   = theta_t - theta_0

    ## define Lambert Problem
    LP = LambertProblem( mu )

    _r_chs_0 = norm( r_chs_0_ECI )
    _r_trg_t = norm( r_trg_t_ECI )

    t1 = 0
    t2 = t1 + t_tof

    ## solve Lambert Problem
    a = LP.solve( _r_chs_0, _r_trg_t, t1, t2, theta )

    return a, r_chs_0_ORP, r_trg_t_ORP


def transfer_impulse( F1, F2, a, r_xxx_x_ORP ):

    pass


if __name__ == "__main__":

    r_chs_0_ECI, v_chs_0_ECI = estimate( O_chs, MU, 0 )

    r_trg_0_ECI, v_trg_0_ECI = estimate( O_trg, MU, 0 )

    r_trg_t_ECI, v_trg_t_ECI = step01( 
        r_trg_0_ECI=r_trg_0_ECI,
        v_trg_0_ECI=v_trg_0_ECI,
        O_trg=O_trg,
        t_tof=t_tof,
        mu=MU
    )

    a, r_chs_0_ORP, r_trg_t_ORP = solve_LP( r_chs_0_ECI, r_trg_t_ECI, t_tof, MU )

    F1, F2 = get_foci_by_a( a, r_chs_0_ORP, r_trg_t_ORP )

    transfer_impulse( F1, F2, a, r_chs_0_ORP )


    pos_chs, vel_chs = elliptic_orbit( O_chs, rev=2 )
    pos_trg, vel_trg = elliptic_orbit( O_trg, rev=2 )

    fig = plt.figure( figsize=( 8,8 ) )
    ax  = plt.axes( projection='3d' )

    ax.plot3D(
        pos_chs[:,0],
        pos_chs[:,1],
        pos_chs[:,2]
    )

    ax.plot3D(
        pos_trg[:,0],
        pos_trg[:,1],
        pos_trg[:,2]
    )

    ax.axis('equal')
    
    ax.set_xlabel( 'x-inertia' )
    ax.set_ylabel( 'y-inertia' )

    plt.show()