from lambert import LP_solver

from two_body_problem import estimate

from optimization import newtonRaphson

from .universal_form import UF_func, UF_grad

from .step01 import step01

from geometry import MU

from numpy import zeros



def initial_guess_with_LP( O_chs, O_trg, t_chs, t_trg, t_tof ):

    ## solve lambert problem
    r_chs_0_ECI, v_chs_0_ECI = estimate( O_chs, MU, t_chs )

    r_trg_0_ECI, v_trg_0_ECI = estimate( O_trg, MU, t_trg )

    r_trg_t_ECI, v_trg_t_ECI = step01( 
        r_trg_0_ECI=r_trg_0_ECI,
        v_trg_0_ECI=v_trg_0_ECI,
        O_trg=O_trg,
        t_tof=t_tof,
        mu=MU
    )

    O_orp, Dv0, Dv1, F1, F2 = LP_solver( 
        r_chs_0_ECI=r_chs_0_ECI,
        v_chs_0_ECI=v_chs_0_ECI,
        r_trg_t_ECI=r_trg_t_ECI,
        v_trg_t_ECI=v_trg_t_ECI,
        t_tof=t_tof,
        mu=MU
    )

    configs = {
        "r_chs_0_ECI": r_chs_0_ECI,
        "v_chs_0_ECI": v_chs_0_ECI,
        "r_trg_0_ECI": r_trg_0_ECI,
        "v_trg_0_ECI": v_trg_0_ECI,
        "trg_a"      : O_trg["a"]
    }

    print( '=' * 50 )
    print( 'lambert solved' )
    print( '=' * 50 )

    ## generate initial guess
    '''
    X = [ x, |x, t, Dv0, Dv1, a ]
    
    x  : universal variable of chaser
    |x : universal variable of target
    t  : t_tof
    Dv0: first impulse
    Dv1: final impulse
    a  : semimajor axis
    '''
    args_chs = {
        "r0": r_chs_0_ECI,
        "v0": v_chs_0_ECI+Dv0,
        "mu": MU,
        "a" : O_orp["a"],
        "t" : t_tof
    }

    args_trg = {
        "r0": r_trg_0_ECI,
        "v0": v_trg_0_ECI,
        "mu": MU,
        "a" : O_trg["a"],
        "t" : t_tof
    }

    chs_func = UF_func( args_chs )
    chs_grad = UF_grad( args_chs )

    trg_func = UF_func( args_trg )
    trg_grad = UF_grad( args_trg )

    x_chs = newtonRaphson( chs_func, chs_grad, 0 )
    x_trg = newtonRaphson( trg_func, trg_grad, 0 )

    X = zeros( 10 )
    X[ 0 ] = x_chs
    X[ 1 ] = x_trg
    X[ 2 ] = t_tof
    X[3:6] = Dv0
    X[6:9] = Dv1
    X[ 9 ] = O_orp["a"]

    return X, configs