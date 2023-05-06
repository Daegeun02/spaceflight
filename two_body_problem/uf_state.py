from optimization import newtonRaphson

from .universal_form import UF_func, UF_grad
from .universal_form import FG_expr



## Universal Formulation F and G State estimate
def UF_FG_S( r_xxx_0_xxx, v_xxx_0_xxx, O_xxx, t_xxx, mu ):

    a = O_xxx["a"]

    configs = {
        "r0": r_xxx_0_xxx,
        "v0": v_xxx_0_xxx,
        "mu": mu,
        "a" : a,
        "t" : t_xxx
    }

    func = UF_func( configs )
    grad = UF_grad( configs )

    x = newtonRaphson( func, grad, 0.0, force_return=True )

    r_xxx_t_xxx, v_xxx_t_xxx = FG_expr( x, r_xxx_0_xxx, v_xxx_0_xxx, configs )

    return r_xxx_t_xxx, v_xxx_t_xxx