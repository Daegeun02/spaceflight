from trajectory import elliptic_orbit

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from numpy import deg2rad



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


if __name__ == "__main__":

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