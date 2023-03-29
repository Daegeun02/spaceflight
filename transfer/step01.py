from universal_form import UF_func, UF_grad
from universal_form import f_and_g_expression

from optimization import newtonRaphson

from numpy import zeros, zeros_like
from numpy import cos, sin



def step_1( r_trg_0_ECI, v_trg_0_ECI, O_trg, t_tof, geometry ):

    mu = geometry.mu

    ## 1. transform coordinate system
    a = O_trg["a"]
    o = O_trg["o"]
    i = O_trg["i"]
    w = O_trg["w"]

    R = PQW2ECI( o, i, w )

    r_trg_0_PQW = R.T @ r_trg_0_ECI
    v_trg_0_PQW = R.T @ v_trg_0_ECI
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

    r_trg_t_ECI = R @ r_trg_t_PQW
    v_trg_t_ECI = R @ v_trg_t_PQW

    return r_trg_t_ECI, v_trg_t_ECI


def PQW2ECI( o, i, w ):

    R = zeros((3,3))
    
    co, so = cos( o ), sin( o )
    ci, si = cos( i ), sin( i )
    cw, sw = cos( w ), sin( w )

    R[0,0] = cw * co - sw * ci * so
    R[1,0] = cw * so + sw * ci * co
    R[2,0] = sw * si
    R[0,1] = (-1) * (sw * co + cw * ci * so)
    R[1,1] = cw * ci * co - sw * so
    R[2,1] = cw * si
    R[0,2] = si * so
    R[1,2] = (-1) * si * co
    R[2,2] = ci

    return R