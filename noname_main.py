from transfer import NN_func, NN_jacb

from transfer import initial_guess_with_LP

from optimization import augmentedLagrangian

from numpy import zeros
from numpy import deg2rad

from numpy.random import rand



def rendezvous( O_chs, O_trg, t_chs, t_trg, t_tof ):

    f = zeros( 10 )
    J = zeros((10,10))

    x0, configs = initial_guess_with_LP( O_chs, O_trg, t_chs, t_trg, t_tof )

    # x0 = zeros( 10 )
    # x0[ 0 ] = 200
    # x0[ 1 ] = 200
    # x0[ 2 ] = 10000
    # x0[3:6] = rand( 3 )
    # x0[6:9] = rand( 3 )
    # x0[ 9 ] = O_trg["a"]

    func = NN_func( configs, f )
    jacb = NN_jacb( configs, J )

    x_opt = augmentedLagrangian( func, jacb, x0, 2, 8, lam=1000 )

    print( x_opt )


## simulation
t_tof = 8000
t_chs = 1000
t_trg = 6000

O_chs = {
    "a": 7000,
    "o": deg2rad( 30 ),
    "i": deg2rad( 0 ),
    "w": deg2rad( 0 ),
    "e": 0.1,
    "T": 0
}

O_trg = {
    "a": 20000,
    "o": deg2rad( 0 ),
    "i": deg2rad( 30 ),
    "w": deg2rad( 30 ),
    "e": 0.5,
    "T": 0
}


if __name__ == "__main__":

    rendezvous( 
        O_chs,
        O_trg,
        t_chs,
        t_trg,
        t_tof
    )