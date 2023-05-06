## lambert problem solver
from two_body_problem import UF_FG_S

from .lambert import LambertProblem

from .focus_calculus import get_foci_by_a_without
from .focus_calculus import get_ECI2PQW_from_foci

from .impulse_control import impulse_ctrl_without

from numpy import dot, cross
from numpy import sqrt
from numpy import arccos, pi

from numpy.linalg import norm



def Build_LP_solver( r_chs_x_ECI, v_chs_x_ECI, r_trg_x_ECI, v_trg_x_ECI, mu, O_chs, O_trg ):
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
    def _func( t ):
        tw, t_tof = t
        ## department
        r_chs_0_ECI, v_chs_0_ECI = UF_FG_S( r_chs_x_ECI, v_chs_x_ECI, O_chs, tw, mu )
        ## destination
        r_trg_t_ECI, v_trg_t_ECI = UF_FG_S( r_trg_x_ECI, v_trg_x_ECI, O_trg, tw+t_tof, mu )

        ## direction of angular momentum vector
        H = cross( r_chs_0_ECI, r_trg_t_ECI )
        h =  H / norm( H )

        ## distance from focus
        r1 = norm( r_chs_0_ECI )
        r2 = norm( r_trg_t_ECI )

        ## angle between two vectors
        theta = arccos( dot( r_chs_0_ECI, r_trg_t_ECI ) / ( r1 * r2 ) )

        ## solve Lambert Problem so calculate semimajor axis
        LP = LambertProblem( )
        a  = LP.solve( r1, r2, 0, t_tof, theta, mu )

        period = 2 * pi * sqrt( ( a**3 ) / mu )

        F1, F2 = get_foci_by_a_without( a, h, r_chs_0_ECI, r_trg_t_ECI )

        ## eccentricity
        ae = norm( F1 )
        e1 = ae / ( 2 * a )

        R1 = get_ECI2PQW_from_foci( F1, h )

        O_orp_F1 = {
            'a': a,
            'e': e1,
            'R': R1
        }

        if ( t_tof > ( period / 2 ) ):
            Dv0_F1 = impulse_ctrl_without( r_chs_0_ECI, v_chs_0_ECI, O_orp_F1, mu, reverse=True )
            Dv1_F1 = impulse_ctrl_without( r_trg_t_ECI, v_trg_t_ECI, O_orp_F1, mu, reverse=True )
        else:
            Dv0_F1 = impulse_ctrl_without( r_chs_0_ECI, v_chs_0_ECI, O_orp_F1, mu )
            Dv1_F1 = impulse_ctrl_without( r_trg_t_ECI, v_trg_t_ECI, O_orp_F1, mu )

        return O_orp_F1, Dv0_F1, -Dv1_F1, F1

        ae = norm( F2 )
        e2 = ae / ( 2 * a )

        R2 = get_ECI2PQW_from_foci( F2, h )

        O_orp_F2 = {
            'a': a,
            'e': e2,
            'R': R2
        }

        if ( t_tof > ( period / 2 ) ):
            Dv0_F2 = impulse_ctrl_without( r_chs_0_ECI, v_chs_0_ECI, O_orp_F2, mu, reverse=True )
            Dv1_F2 = impulse_ctrl_without( r_trg_t_ECI, v_trg_t_ECI, O_orp_F2, mu, reverse=True )
        else:
            Dv0_F2 = impulse_ctrl_without( r_chs_0_ECI, v_chs_0_ECI, O_orp_F2, mu )
            Dv1_F2 = impulse_ctrl_without( r_trg_t_ECI, v_trg_t_ECI, O_orp_F2, mu )

        Dv__F1 = norm( Dv0_F1 ) + norm( Dv1_F1 )
        Dv__F2 = norm( Dv0_F2 ) + norm( Dv1_F2 )

        if ( Dv__F1 < Dv__F2 ):

            return ( O_orp_F1, Dv0_F1, -Dv1_F1, F1 )

        else:

            return ( O_orp_F2, Dv0_F2, -Dv1_F2, F2 )

    return _func