from JAX_universal import UF_func, UF_grad
from JAX_universal import FG_expr

from JAX_lambert import *

from JAX_optimizer import newtonRaphson
from JAX_optimizer import levenbergMarquardt

from jax import grad

from jax.numpy import array
from jax.numpy import dot, cross
from jax.numpy import sqrt, pi
from jax.numpy import cos, arccos
from jax.numpy import sin

from jax.numpy.linalg import norm



def forward( configs ):

    r_chs_0_ECI = configs["r_chs_0_ECI"]
    v_chs_0_ECI = configs["v_chs_0_ECI"]
    r_trg_0_ECI = configs["r_trg_0_ECI"]
    v_trg_0_ECI = configs["v_trg_0_ECI"]

    mu = configs["mu"]
    t  = configs["t"]

    a_chs = configs["a_chs"]
    a_trg = configs["a_trg"]

    def _func( t ):

        chs_configs = {
            "r0": r_chs_0_ECI,
            "v0": v_chs_0_ECI,
            "mu": mu,
            "a" : a_chs,
            "t" : t
        }

        trg_configs = {
            "r0": r_trg_0_ECI,
            "v0": v_trg_0_ECI,
            "mu": mu,
            "a" : a_trg,
            "t" : t
        }

        _UF_func = UF_func( trg_configs )
        _UF_grad = UF_grad( trg_configs )
        _FG_func = FG_expr( trg_configs )

        x = newtonRaphson( _UF_func, _UF_grad, 0.0, force_return=True )

        _r_v        = _FG_func( x, t )
        r_trg_t_ECI = _r_v[0:3]
        v_trg_t_ECI = _r_v[3:6]

        H = cross( r_chs_0_ECI, r_trg_t_ECI )
        h = H / norm( H )

        r1 = norm( r_chs_0_ECI )
        r2 = norm( r_trg_t_ECI )

        theta = arccos( dot( r_chs_0_ECI, r_trg_t_ECI ) / ( r1 * r2 ) )

        c = sqrt( r1**2 + r2**2 - 2 * dot( r_chs_0_ECI, r_trg_t_ECI ) )
        s = (0.5) * ( r1 + r2 + c )

        q = ( sqrt( r1 * r2 ) / s ) * cos( theta/2 )

        T = sqrt( ( 8 * mu ) / ( s**3 ) ) * t

        _LP_func = LP_func( T, q )
        _LP_jacb = LP_jacb( T, q )

        init_params1 = array([pi, 0])
        xS1 = levenbergMarquardt( _LP_func, _LP_jacb, init_params1, force_return=True )

        _a = s / ( 1 - cos( xS1[0] ) )

        period = 2 * pi * sqrt( ( _a**3 ) / mu )

        _r0 = norm( r_chs_0_ECI - r_trg_t_ECI )
        _r1 = 2 * _a - r1
        _r2 = 2 * _a - r2

        _FA_func = FA_func( _r0, _r1, _r2 )
        _FA_jacb = FA_jacb( _r0, _r1, _r2 )

        init_params2 = array([pi/4, pi/4])
        xS2 = levenbergMarquardt( _FA_func, _FA_jacb, init_params2, force_return=True )

        l = _r1 * sin( xS2[0] )
        
        d = (r_trg_t_ECI - r_chs_0_ECI) / norm( r_trg_t_ECI - r_chs_0_ECI )

        O = r_chs_0_ECI + sqrt( _r1**2 - l**2 ) * d

        D = cross( h, d )

        F1 = O + D * l
        F2 = O - D * l

        e1 = norm( F1 ) / ( 2 * _a )
        e2 = norm( F2 ) / ( 2 * _a )

        Q1 = cross( h, F1 )
        Q2 = cross( h, F2 )

        R1 = array([
            -F1 / norm( F1 ),
            -Q1 / norm( Q1 ),
            h
        ])

        R2 = array([
            -F2 / norm( F2 ),
            -Q2 / norm( Q2 ),
            h
        ])

        p1 = _a * ( 1 - e1**2 )
        p2 = _a * ( 1 - e2**2 )
        C1 = sqrt( mu / p1 )
        C2 = sqrt( mu / p2 )

        ## Dv for Focus 1 BEGIN ##
        r_chs_0_PQW = R1 @ r_chs_0_ECI
        cos_f = r_chs_0_PQW[0] / r1
        sin_f = r_chs_0_PQW[1] / r1

        v_chs_0_PQW = array([
            (-C1) * sin_f,
            C1 * ( e1 + cos_f ),
            0
        ])

        # r_trg_t_PQW = R1 @ r_trg_t_ECI
        # cos_f = r_trg_t_PQW[0] / r2
        # sin_f = r_trg_t_PQW[1] / r2

        # v_trg_t_PQW = array([
        #     -(C1) * sin_f,
        #     C1 * ( e1 + cos_f ),
        #     0
        # ])

        if ( t < ( period / 2 ) ):
            Dv0_F1 = ( R1.T @ v_chs_0_PQW ) - v_chs_0_ECI
            # Dv1_F1 = ( R1.T @ v_trg_t_PQW ) - v_trg_t_ECI
        else:
            Dv0_F1 = - ( ( R1.T @ v_chs_0_PQW ) + v_chs_0_ECI )
            # Dv1_F1 = - ( ( R1.T @ v_trg_t_PQW ) + v_trg_t_ECI )
        ## Dv for Focus 1 END ##

        ## Dv for Focus 2 BEGIN ##
        r_chs_0_PQW = R2 @ r_chs_0_ECI
        cos_f = r_chs_0_PQW[0] / r1
        sin_f = r_chs_0_PQW[1] / r1

        v_chs_0_PQW = array([
            (-C2) * sin_f,
            C2 * ( e2 + cos_f ),
            0
        ])

        # r_trg_t_PQW = R2 @ r_trg_t_ECI
        # cos_f = r_trg_t_PQW[0] / r2
        # sin_f = r_trg_t_PQW[1] / r2

        # v_trg_t_PQW = array([
        #     -(C2) * sin_f,
        #     C2 * ( e2 + cos_f ),
        #     0
        # ])

        if ( t < ( period / 2 ) ):
            Dv0_F2 = ( R2.T @ v_chs_0_PQW ) - v_chs_0_ECI
            # Dv1_F2 = ( R2.T @ v_trg_t_PQW ) - v_trg_t_ECI
        else:
            Dv0_F2 = - ( ( R2.T @ v_chs_0_PQW ) + v_chs_0_ECI )
            # Dv1_F2 = - ( ( R2.T @ v_trg_t_PQW ) + v_trg_t_ECI )
        ## Dv for Focus 2 END ##

        Dv_F1 = norm( Dv0_F1 ) # + norm( Dv1_F1 )
        Dv_F2 = norm( Dv0_F2 ) # + norm( Dv1_F2 )

        if ( Dv_F1 < Dv_F2 ):
            return Dv_F1
        else:
            return Dv_F2
   
    return _func



if __name__ == "__main__":
    print( '<<< JAX >>>' )

    GRAVCONST = 6.674e-11
    EARTHMASS = 5.9742e24
    MU        = GRAVCONST * EARTHMASS / ( 1000 ** 3 )

    r_chs_0_ECI = array( [2594.76851273, 8860.1442633,     0.        ] )
    v_chs_0_ECI = array( [-6.40773121,  2.33323034,  0.        ] )
    r_trg_0_ECI = array( [-18966.02307806,   1460.66653646,    843.31621802] )
    v_trg_0_ECI = array( [-2.01566097, -2.90294796, -1.67601778] )

    mu = MU
    t  = 9000.0

    a_chs = 10000.0
    a_trg = 15000.0

    configs = {
        "r_chs_0_ECI": r_chs_0_ECI,
        "v_chs_0_ECI": v_chs_0_ECI,
        "r_trg_0_ECI": r_trg_0_ECI,
        "v_trg_0_ECI": v_trg_0_ECI,

        "mu": mu,
        "t" : t,
        
        "a_chs": a_chs,
        "a_trg": a_trg
    }

    _func = forward( configs )

    val = _func( t )

    _grad = grad( _func )

    print( 'val =>', val )

    print( 'dot =>', array( _grad( t ) ) )