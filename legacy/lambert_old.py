from .focus_calculus import get_foci_by_a, get_elem_by_foci

from .impulse_control import impulse_ctrl

from numpy import cross, sqrt
from numpy import arctan2, pi

from numpy.linalg import norm



def LP_solver( r_chs_0_ECI, v_chs_0_ECI, r_trg_t_ECI, v_trg_t_ECI, t_tof, mu ):
    '''
    Calculate transfer orbit from r1 to r2.

    This function solve root finding problem 
    to get two major factor of transfer orbit.

    Solve root finding with Levenberg-Marquardt Algorithm

    From two major factor of transfer orbit,
    calculate six elements of orbit.

    parameter 
    r1: vector from focus to starting point 
    r2: vector from focus to arriving point
    t1: time when starting transfer
    t2: time when arriving transfer
    theta: the angle between r1 and r2 vector
    '''
    ## direction of angular momentum vector
    H = cross( r_chs_0_ECI, r_trg_t_ECI )
    h =  H / norm( H )

    ## define orbital plane
    o = arctan2(          h[0], h[1] ) * ( -1 )
    i = arctan2( norm( h[:2] ), h[2] ) * ( -1 )

    ## distance from focus
    r1 = norm( r_chs_0_ECI )
    r2 = norm( r_trg_t_ECI )

    R = ECI2ORP( o, i )
    r_chs_0_ORP = R @ r_chs_0_ECI
    r_trg_t_ORP = R @ r_trg_t_ECI

    ## angle between two vectors
    N_chs_ORP = arctan2( r_chs_0_ORP[1], r_chs_0_ORP[0] )
    N_trg_ORP = arctan2( r_trg_t_ORP[1], r_trg_t_ORP[0] )
    theta     = N_trg_ORP - N_chs_ORP

    ## solve Lambert Problem so calculate semimajor axis
    LP = LambertProblem( )
    a  = LP.solve( r1, r2, 0, t_tof, theta, mu )

    F1, F2 = get_foci_by_a( a, r_chs_0_ORP, r_trg_t_ORP )

    O_orp = {
        'a': a,
        'e': 0,
        'o': o,
        'i': i,
        'w': 0,
        'T': 0
    }

    O_orp_F1 = dict( O_orp )
    O_orp_F2 = dict( O_orp )

    period = 2 * pi * sqrt( ( a**3 ) / mu )

    ## recalculate orbital element
    get_elem_by_foci( F1, O_orp_F1 )
    get_elem_by_foci( F2, O_orp_F2 )
    ## ORP to ECI
    F1 = R.T @ F1
    F2 = R.T @ F2

    if ( t_tof > ( period / 2 ) ):
        Dv0_F1 = impulse_ctrl( r_chs_0_ECI, v_chs_0_ECI, O_orp_F1, mu, reverse=True )
        Dv1_F1 = impulse_ctrl( r_trg_t_ECI, v_trg_t_ECI, O_orp_F1, mu, reverse=True )
    else:
        Dv0_F1 = impulse_ctrl( r_chs_0_ECI, v_chs_0_ECI, O_orp_F1, mu )
        Dv1_F1 = impulse_ctrl( r_trg_t_ECI, v_trg_t_ECI, O_orp_F1, mu )

    if ( t_tof > ( period / 2 ) ):
        Dv0_F2 = impulse_ctrl( r_chs_0_ECI, v_chs_0_ECI, O_orp_F2, mu, reverse=True )
        Dv1_F2 = impulse_ctrl( r_trg_t_ECI, v_trg_t_ECI, O_orp_F2, mu, reverse=True )
    else:
        Dv0_F2 = impulse_ctrl( r_chs_0_ECI, v_chs_0_ECI, O_orp_F2, mu )
        Dv1_F2 = impulse_ctrl( r_trg_t_ECI, v_trg_t_ECI, O_orp_F2, mu )

    Dv__F1 = norm( Dv0_F1 ) + norm( Dv1_F1 )
    Dv__F2 = norm( Dv0_F2 ) + norm( Dv1_F2 )

    if ( Dv__F1 < Dv__F2 ):

        return O_orp_F1, Dv0_F1, -Dv1_F1, F1

    else:

        return O_orp_F2, Dv0_F2, -Dv1_F2, F2 