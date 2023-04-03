from trajectory import elliptic_orbit

from lambert import LP_solver

from two_body_problem import estimate
from two_body_problem import estimate_with_foci

from transfer import step01

from geometry import MU

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from numpy import deg2rad

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

    LP_solver( r_chs_0_ECI, r_trg_t_ECI, t_tof, MU )

    # O_orp, r_chs_0_ORP, r_trg_t_ORP = solve_LP( r_chs_0_ECI, r_trg_t_ECI, t_tof, MU )

    raise ValueError

    a = O_orp['a']

    F1, F2 = get_foci_by_a( a, r_chs_0_ORP, r_trg_t_ORP )

    ae = norm( F2 - F1 )
    e  = ae / a
    O_orp['e'] = e

    Dv0 = transfer_impulse( F1, F2, a, r_chs_0_ORP )
    Dv1 = transfer_impulse( F1, F2, a, r_trg_t_ORP )

    impulse = {
        0    : Dv0,
        t_tof: Dv1
    }

    pos_chs, vel_chs = elliptic_orbit( O_chs, rev=2 )
    pos_orp, vel_orp = elliptic_orbit( O_orp, rev=2, impulse=impulse)
    pos_trg, vel_trg = elliptic_orbit( O_trg, rev=2 )

    fig = plt.figure( figsize=( 8,8 ) )
    ax  = plt.axes( projection='3d' )

    ax.plot3D(
        pos_chs[:,0],
        pos_chs[:,1],
        pos_chs[:,2]
    )

    ax.plot3D(
        pos_orp[:,0],
        pos_orp[:,1],
        pos_orp[:,2]
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