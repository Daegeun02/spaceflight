from jax import numpy 

from JAX_universal import UF_func
from JAX_universal import UF_back
from JAX_universal import FG_expr

from JAX_optimizer import newtonRaphson



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