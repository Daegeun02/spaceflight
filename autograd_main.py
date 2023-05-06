from lambert import Build_LP_solver

from two_body_problem import estimate

from geometry import MU

from numpy import linspace
from numpy import zeros
from numpy import deg2rad
from numpy import int64
from numpy import save

from numpy.linalg import norm

import matplotlib.pyplot as plt

import proplot as pplt

from pandas import read_csv



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

t_chs = 000.0
t_trg = 000.0

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
        t = [0.0, t_tof]
        O_orp, Dv0, Dv1, F = _solver( t )
        Dv0s.append( norm( Dv0 ) )

    plt.plot( t_tofs, Dv0s, label="impulse wave" )

    plt.title( 'velocity impulse for each time of flight via lambert solution')
    plt.xlabel( 'time of flight' )
    plt.ylabel( 'magnitude of Dv0 [Km/s]' )

    plt.xlim([t_tofs[0],t_tofs[-1]])
    plt.ylim([0,16])

    plt.legend()
    plt.grid()

    plt.show()


def multi():

    tws = linspace(-5000,20000,101)
    t1s = linspace(-5000,20000,101)

    Dvs = zeros( (len(tws), len(t1s)) )

    for i in range( len( tws ) ):
        for j in range( len( t1s ) ):
            t = [tws[i], t1s[j]]

            O_orp, Dv0, Dv1, F = _solver( t )
            Dvs[i,j] = norm( Dv0 )
    
    save( './data/tws_3', tws )
    save( './data/t1s_3', t1s )
    save( './data/Dvs_3', Dvs )

    df = read_csv( './data/gradient_memory.csv' )
    tws_trace = df['Unnamed: 1']
    t1s_trace = df['Unnamed: 2']
    Dvs_trace = df['Unnamed: 3']

    tws = load( './data/tws_3.npy' )
    t1s = load( './data/t1s_3.npy' )
    Dvs = load( './data/Dvs_3.npy' )

    lim = (-5000,20000)
        
    fig = pplt.figure( refwidth=2.3, share=False )
    axs = fig.subplots(ncols=1, nrows=1)
    axs.format(
        xlabel='wait time', ylabel='time of flight',
        xlim=lim, ylim=lim,
        suptitle='velocity impulse via lambert problem'
    )

    axs[0].contourf( tws, t1s, Dvs, colorbar='b' )
    axs[0].scatter( tws_trace, t1s_trace, color='g' )
    axs[0].plot( tws_trace, t1s_trace, color='y' )

    pplt.show()


if __name__ == "__main__":
    from numpy import load

    single()
    # multi()