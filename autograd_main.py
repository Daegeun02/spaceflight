from lambert import Build_LP_solver

from two_body_problem import estimate

from geometry import MU

from numpy import linspace
from numpy import zeros
from numpy import deg2rad
from numpy import int64

from numpy.linalg import norm

import matplotlib.pyplot as plt



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

t_chs = 0
t_trg = 0

## estimate chaser's position and velocity
r_chs_0_ECI, v_chs_0_ECI = estimate( O_chs, MU, t_chs )
## estimate target's initial position and velocity
r_trg_0_ECI, v_trg_0_ECI = estimate( O_trg, MU, t_trg )

_solver = Build_LP_solver(
    r_chs_0_ECI, v_chs_0_ECI, r_trg_0_ECI, v_trg_0_ECI, 
    MU, O_chs, O_trg
)

def single():

    t_tofs = linspace(9000,23000,701).astype(int64)

    Dv0s = []
    for t_tof in t_tofs:
        O_orp, Dv0, Dv1, F = _solver( t_tof )
        Dv0s.append( norm( Dv0 ) )

    plt.plot( t_tofs, Dv0s, label="impulse wave" )

    plt.title( 'velocity impulse for each time of flight via lambert solution')
    plt.xlabel( 'time of flight' )
    plt.ylabel( 'magnitude of Dv0 [Km/s]' )

    plt.legend()
    plt.grid()

    plt.show()


def multi():

    tws = linspace(-5000,20000,251).astype(int64)
    t1s = linspace(-5000,20000,251).astype(int64)

    Dv0s = zeros( (len(tws), len(t1s)) )

    for i in range( len( tws ) ):
        for j in range( len( t1s ) ):
            tw = tws[i]
            t1 = t1s[j]

            O_orp, Dv0, Dv1, F = _solver( t1, tw )
            Dv0s[i,j] = Dv0
        


if __name__ == "__main__":

    multi()