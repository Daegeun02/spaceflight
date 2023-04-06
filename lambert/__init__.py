## lambert problem solver
from coordinate import ECI2ORP
from coordinate import ECI2PQW

from .lambert import LambertProblem

from .focus_calculus import get_foci_by_a, get_elem_by_foci

from .impulse_control import impulse_ctrl

from numpy import cross, dot
from numpy import arctan2
from numpy import arccos

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
    h = H / norm( H )

    ## define orbital plane
    o = arctan2(          h[0], h[1] ) * ( -1 )
    i = arctan2( norm( h[:2] ), h[2] ) * ( -1 )

    ## distance from focus
    r1 = norm( r_chs_0_ECI )
    r2 = norm( r_trg_t_ECI )

    ## angle between two vectors
    value = dot( r_chs_0_ECI, r_trg_t_ECI ) / ( r1 * r2 )
    theta = arccos( value )

    ## solve Lambert Problem so calculate semimajor axis
    LP = LambertProblem( )
    a  = LP.solve( r1, r2, 0, t_tof, theta, mu )

    print( a )

    R = ECI2ORP( o, i )
    r_chs_0_ORP = R @ r_chs_0_ECI
    r_trg_t_ORP = R @ r_trg_t_ECI

    F1, F2 = get_foci_by_a( a, r_chs_0_ORP, r_trg_t_ORP )

    O_orp = {
        'a': a,
        'e': 0,
        'o': o,
        'i': i,
        'w': 0,
        'T': 0
    }
    ## recalculate orbital element
    get_elem_by_foci( F1, O_orp )
    ## ORP to ECI
    F1 = R.T @ F1
    F2 = R.T @ F2

    ## entry burn
    Dv0 = impulse_ctrl( r_chs_0_ECI, v_chs_0_ECI, O_orp, mu )
    ## geton burn
    Dv1 = impulse_ctrl( r_trg_t_ECI, v_trg_t_ECI, O_orp, mu )

    print( Dv0, Dv1 )
    print( O_orp )

    return O_orp, Dv0, -Dv1, F1, F2 