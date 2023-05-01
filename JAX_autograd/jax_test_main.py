from JAX_universal import UF_func
from JAX_universal import UF_back
from JAX_universal import FG_expr
from JAX_universal import FG_back

from JAX_optimizer import newtonRaphson

from jax.numpy import array
from jax.numpy import dot
from jax.numpy import sqrt, cos, sin

from jax.numpy.linalg import norm

import numpy as np



GRAVCONST = 6.674e-11

EARTHMASS = 5.9742e24

MU = GRAVCONST * EARTHMASS / ( 1000 ** 3 )

r_chs_0_ECI = array( [-1449.26847138, 6569.08693194, 0.        ] )
v_chs_0_ECI = array( [-7.78632738, -0.97724683, 0.        ] )
r_trg_0_ECI = array( [-12944.83921337, -6562.56493754, -3788.89863326] )
v_trg_0_ECI = array( [1.86069651, -2.71514808, -1.56759148] )

mu = MU
a  = 10000.0
t  = 4000.0


def analytic_backward( configs, t ):

    configs = {
        "r_chs_0_ECI": r_chs_0_ECI,
        "v_chs_0_ECI": v_chs_0_ECI,
        "r_trg_0_ECI": r_trg_0_ECI,
        "v_trg_0_ECI": v_trg_0_ECI,

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

    r0 = r_trg_0_ECI
    v0 = v_trg_0_ECI

    _UF_func = UF_func( trg_configs )
    _UF_grad = UF_back( trg_configs )

    x = newtonRaphson( _UF_func, 0.0 ) 

    r = 1
    r += ( dot( r0, v0 ) / sqrt( mu * a ) ) * sin( x / sqrt( a ) )
    r += ( ( norm( r0 ) / a ) - 1 ) * cos( x / sqrt( a ) )
    r *= a 



if __name__ == "__main__":

    print( 'xdot', sqrt( mu ) / r )

    _FG_func = FG_expr( configs )
    _FG_jacb = FG_back( configs )

    rv   = _FG_func( x, t )
    prp_ = np.array( _FG_jacb( x, t ) ).T.reshape(-1,2)

    prpt = prp_[0:3,0] * _UF_grad( t ) + prp_[0:3,1] 

    print( rv )
    # print( prp_ )
    print( prpt )

    pfpx = (-1) * ( a / norm( r0 ) ) * sin( x / sqrt( a ) ) * ( 1 / sqrt( a ) )
    pgpx = ( a / sqrt( mu ) ) * ( cos( x / sqrt( a ) ) - 1 )

    # print( pfpx * r0[:] + pgpx * v0[:] )

    # print( prpt - ( pfpx * r0[:] * _UF_grad( t ) + pgpx * v0[:] * _UF_grad( t ) ) )

    print( '===' * 20 )

    f = 1 - ( a / norm( r0 ) ) * ( 1 - cos( x / sqrt( a ) ) )
    g = t - ( a / sqrt( mu ) ) * ( x - sqrt( a ) * sin( x / sqrt( a ) ) )

    rt = f * r0 + g * v0

    print( 'rt', norm( rt ) )

    print( 'r', r )

    fdot = (-1) * ( ( sqrt( a ) * sqrt( mu ) ) / ( norm( r0 ) * r ) ) * sin( x / sqrt( a ) )
    gdot = 1 - ( a / r ) * ( 1 - cos( x / sqrt( a ) ) )

    vt = fdot * r0 + gdot * v0

    print( 'vt', vt )