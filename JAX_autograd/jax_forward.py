from JAX_universal import UF_func
from JAX_universal import FG_expr

from JAX_lambert import AM_func
from JAX_lambert import LambertProblem

from JAX_optimizer import newtonRaphson

from jax import jacfwd

from jax.numpy import array, zeros
from jax.numpy import dot
from jax.numpy import sqrt, pi
from jax.numpy import cos, arccos
from jax.numpy import sin

from jax.numpy.linalg import norm

from jaxopt import LevenbergMarquardt



def forward( configs ):

    r_chs_0_ECI = configs["r_chs_0_ECI"]
    v_chs_0_ECI = configs["v_chs_0_ECI"]
    r_trg_0_ECI = configs["r_trg_0_ECI"]
    v_trg_0_ECI = configs["v_trg_0_ECI"]

    mu = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    def _func( t ):

        chs_configs = {
            "r0": r_chs_0_ECI,
            "v0": v_chs_0_ECI,
            "mu": mu,
            "a" : a,
            "t" : t
        }

        trg_configs = {
            "r0": r_trg_0_ECI,
            "v0": v_trg_0_ECI,
            "mu": mu,
            "a" : a,
            "t" : t
        }

        configs["t"] = t

        _UF_func = UF_func( trg_configs )
        _FG_func = FG_expr( trg_configs )
        _AM_func = AM_func( configs )

        x = newtonRaphson( _UF_func, 0.0 )

        _r_v        = _FG_func( x, t )
        r_trg_t_ECI = _r_v[0:3]
        v_trg_t_ECI = _r_v[3:6]

        h = _AM_func( _r_v[0:3] )

        r1 = norm( r_chs_0_ECI )
        r2 = norm( r_trg_t_ECI )

        theta = arccos( dot( r_chs_0_ECI, r_trg_t_ECI ) / ( r1 * r2 ) )

        c = sqrt( r1**2 + r2**2 - 2 * dot( r_chs_0_ECI, r_trg_t_ECI ) )
        s = (0.5) * ( r1 + r2 + c )

        q = ( sqrt( r1 * r2 ) / s ) * cos( theta/2 )

        T = sqrt( ( 8 * mu ) / ( s**3 ) ) * t

        return s, q, T
   
    return _func


def LP_back( s, q, T ):

    x0 = array([ pi, 0.0 ])
    
    LM = LevenbergMarquardt( LP_func )
    xS = LM.run( x0, q, T ).params

    _a = s / ( 1 - cos( xS[0] ) )

    return _a
        

def LP_func( s, q, T ):

    def _func( x ):
        a, b = x
        out  = zeros(2)

        out = out.at[0].set(
            sin( b / 2 ) - q * sin( a / 2 ),
        )
        out = out.at[1].set(
            T * ( sin( a / 2 ) ** 3 ) - ( a - b - sin( a ) + sin( b ) )
        )

        return out

    x0 = array([ pi, 0.0 ])
    
    LM = LevenbergMarquardt( _func )
    xS = LM.run( x0 ).params

    _a = s / ( 1 - cos( xS[0] ) )

    return _a



if __name__ == "__main__":
    print( '<<< JAX >>>' )

    GRAVCONST = 6.674e-11
    EARTHMASS = 5.9742e24
    MU        = GRAVCONST * EARTHMASS / ( 1000 ** 3 )

    r_chs_0_ECI = array( [-1449.26847138, 6569.08693194, 0.        ] )
    v_chs_0_ECI = array( [-7.78632738, -0.97724683, 0.        ] )
    r_trg_0_ECI = array( [-12944.83921337, -6562.56493754, -3788.89863326] )
    v_trg_0_ECI = array( [1.86069651, -2.71514808, -1.56759148] )

    mu = MU
    a  = 10000.0
    t  = 3900.0

    configs = {
        "r_chs_0_ECI": r_chs_0_ECI,
        "v_chs_0_ECI": v_chs_0_ECI,
        "r_trg_0_ECI": r_trg_0_ECI,
        "v_trg_0_ECI": v_trg_0_ECI,

        "mu": mu,
        "a" : a,
        "t" : t
    }

    _func = forward( configs )

    s, q, T = _func( t )

    _a = LP_func( s, q, T )

    print( 'T    =>', s, q, T )
    print( 'a    => ', _a )

    _grad = jacfwd( _func )
    print( 'Tdot =>', array( _grad( t ) ) )

    # _grad = jacfwd( LP_back )
    # print( 'adot => ', _grad( s, q, T ) )