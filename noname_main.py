from transfer import NN_func, NN_jacb

from optimization import augmentedLagrangian

from two_body_problem import estimate

from geometry import MU

from numpy import zeros
from numpy import deg2rad

from numpy.random import rand



def rendezvous( configs ):

    f = zeros( 10 )
    J = zeros((10,10))

    func = NN_func( configs, f )
    jacb = NN_jacb( configs, J )

    x0 = zeros( 10 )
    x0[ 0 ] = 10
    x0[ 1 ] = 20
    x0[ 2 ] = 3000
    x0[3:6] = rand( 3 )
    x0[6:9] = rand( 3 )
    x0[ 9 ] = configs["trg_a"]

    x_opt = augmentedLagrangian( func, jacb, x0, 2, 8, lam=10000 )

    print( x_opt )

O_chs = {
    "a": 7000,
    "o": deg2rad( 0 ),
    "i": deg2rad( 0 ),
    "w": deg2rad( 0 ),
    "e": 0,
    "T": 0
}

O_trg = {
    "a": 10000,
    "o": deg2rad( 0 ),
    "i": deg2rad( 0 ),
    "w": deg2rad( 0 ),
    "e": 0,
    "T": 0
}


if __name__ == "__main__":

    r_chs_0_ECI, v_chs_0_ECI = estimate( O_chs, MU, 0 )

    r_trg_0_ECI, v_trg_0_ECI = estimate( O_trg, MU, 0 )

    configs = {
        "r_trg_0_ECI": r_trg_0_ECI,
        "v_trg_0_ECI": v_trg_0_ECI,
        "r_chs_0_ECI": r_chs_0_ECI,
        "v_chs_0_ECI": v_chs_0_ECI,
        "trg_a"      : O_trg["a"]
    }

    rendezvous( configs )