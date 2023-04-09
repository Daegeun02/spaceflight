from optimization import newtonRaphson

from coordinate import ECI2PQW

from .universal_form import UF_func, UF_grad
from .universal_form import f_and_g_expression

from numpy import zeros_like



def step01( r_trg_0_ECI, v_trg_0_ECI, O_trg, t_tof, mu ):

    ## 1. transform coordinate system
    a = O_trg["a"]
    o = O_trg["o"]
    i = O_trg["i"]
    w = O_trg["w"]

    R = ECI2PQW( o, i, w )

    r_trg_0_PQW = R @ r_trg_0_ECI
    v_trg_0_PQW = R @ v_trg_0_ECI
    r_trg_t_PQW = zeros_like( r_trg_0_PQW )
    v_trg_t_PQW = zeros_like( v_trg_0_PQW )

    args = {
        "r0": r_trg_0_PQW,
        "v0": v_trg_0_PQW,
        "mu": mu,
        "a" : a,
        "t" : t_tof
    }

    func = UF_func( args )
    grad = UF_grad( args )

    x = newtonRaphson( func, grad, 0 )

    r_trg_t_PQW, v_trg_t_PQW = f_and_g_expression( x, r_trg_0_PQW, v_trg_0_PQW, args )

    r_trg_t_ECI = R.T @ r_trg_t_PQW
    v_trg_t_ECI = R.T @ v_trg_t_PQW

    return r_trg_t_ECI, v_trg_t_ECI