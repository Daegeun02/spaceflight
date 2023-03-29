from Opt_MV_based_LP import solve

from numpy import zeros



def step_3( r_chs_0_ORP, v_chs_0_ORP, r_trg_t_ORP, v_trg_t_ORP, theta_t, theta_0, a, geometry ):

    args = {
        "mu": geometry.mu,
        "a" : a,
        "theta_t": theta_t,
        "theta_0": theta_0,

        "r_trg": r_trg_t_ORP,
        "v_trg": v_trg_t_ORP,
        "r_chs": zeros(3),
        "v_chs": zeros(3),
        "r_ch0": r_chs_0_ORP,
        "v_ch0": v_chs_0_ORP,

        "tri_t": zeros(2),
        "tri_0": zeros(2),
        "tri_d": zeros(2),
        
        "fandg": zeros(4)
    }

    x = solve( args )

    Dv0, Dv1, e, w = x

    return Dv0, Dv1, e, w