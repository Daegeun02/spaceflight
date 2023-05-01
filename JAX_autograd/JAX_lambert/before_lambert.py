from jax import numpy 

from JAX_universal import UF_func
from JAX_universal import UF_back
from JAX_universal import FG_expr

from JAX_optimizer import newtonRaphson

from jax.numpy import array

from jax.numpy.linalg import norm



def AM_func( configs ):

    r_chs_0_ECI = configs["r_chs_0_ECI"]
    v_chs_0_ECI = configs["v_chs_0_ECI"]
    r_trg_0_ECI = configs["r_trg_0_ECI"]
    v_trg_0_ECI = configs["v_trg_0_ECI"]

    mu = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    def _func( r_trg_t_ECI ):

        r_SKM = array([
            [0, -r_chs_0_ECI[2], r_chs_0_ECI[1]],
            [r_chs_0_ECI[2], 0, -r_chs_0_ECI[0]],
            [-r_chs_0_ECI[1], r_chs_0_ECI[0], 0]
        ])

        H = r_SKM @ r_trg_t_ECI

        h = H / norm( H )

        return h

    return _func


def UF_FG_S( r_xxx_0_xxx, v_xxx_0_xxx, O_xxx, t_xxx, mu ):

    a = O_xxx["a"]

    configs = {
        "r0": r_xxx_0_xxx,
        "v0": v_xxx_0_xxx,
        "mu": mu,
        "a" : a,
        "t" : t_xxx
    }

    _func = UF_func( configs )

    x = newtonRaphson( _func, 0.0 )

    r_xxx_t_xxx, v_xxx_t_xxx = FG_expr( x, r_xxx_0_xxx, v_xxx_0_xxx, configs )

    return r_xxx_t_xxx, v_xxx_t_xxx