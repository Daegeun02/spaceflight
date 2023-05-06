from geometry import MU

from lambert import Build_LP_solver

from two_body_problem import UF_FG_S

from numpy.linalg import norm



def __init__( configs, init_params ):

    r_chs_0_ECI = configs["r_chs_0_ECI"]
    v_chs_0_ECI = configs["v_chs_0_ECI"]
    r_trg_0_ECI = configs["r_trg_0_ECI"]
    v_trg_0_ECI = configs["v_trg_0_ECI"]

    mu = configs["mu"]

    O_chs = configs["O_chs"]
    O_trg = configs["O_trg"]

    _solver = Build_LP_solver(
        r_chs_0_ECI, v_chs_0_ECI, r_trg_0_ECI, v_trg_0_ECI,
        MU, O_chs, O_trg
    )

    r_wpt_0_ECI, v_wpt_0_ECI, O_wpt = _init_wpt( 
        _solver, init_params
    )

    configs["r_wpt_0_ECI"] = r_wpt_0_ECI
    configs["v_wpt_0_ECI"] = v_wpt_0_ECI
    configs["O_wpt"]       = O_wpt


def solver( configs ):

    r_chs_0_ECI = configs["r_chs_0_ECI"]
    v_chs_0_ECI = configs["v_chs_0_ECI"]
    r_trg_0_ECI = configs["r_trg_0_ECI"]
    v_trg_0_ECI = configs["v_trg_0_ECI"]

    mu = configs["mu"]

    O_chs = configs["O_chs"]
    O_trg = configs["O_trg"]

    def func( t ):
        r_wpt_0_ECI = configs["r_wpt_0_ECI"]
        v_wpt_0_ECI = configs["v_wpt_0_ECI"]
        O_wpt       = configs["O_wpt"]

        tw, t1, t2 = t
        T1 = [tw, t1]
        T2 = [.0, t2]

        r_chs_0_ECI, v_chs_0_ECI = UF_FG_S(
            r_chs_0_ECI, v_chs_0_ECI, O_chs, tw, MU
        )

        r_wpt_0_ECI, v_wpt_0_ECI = UF_FG_S(
            r_wpt_0_ECI, v_wpt_0_ECI, O_wpt, t1, MU
        )

        r_trg_t_ECI, v_trg_t_ECI = UF_FG_S(
            r_trg_0_ECI, v_trg_0_ECI, O_trg, tw+t1+t2, MU
        )

        _solver1 = Build_LP_solver(
            r_chs_0_ECI, v_chs_0_ECI, r_wpt_0_ECI, v_wpt_0_ECI,
            MU, O_chs, O_wpt
        )
        _solver2 = Build_LP_solver(
            r_wpt_0_ECI, v_wpt_0_ECI, r_trg_t_ECI, v_trg_t_ECI,
            MU, O_wpt, O_trg
        )

        _O_wpt, _Dv0, _Dv1, F1 = _solver1( T1 )
        _O_wpt, _Dv2, _Dv3, F2 = _solver2( T2 )

        _Dv = norm( _Dv0 ) + norm( _Dv2 )

        configs["r_wpt_0_ECI"][:] = r_wpt_0_ECI
        configs["v_wpt_0_ECI"][:] = v_wpt_0_ECI
        configs["O_wpt"]['a']     = O_wpt['a']
        configs["O_wpt"]['e']     = O_wpt['e']
        configs["O_wpt"]['R']     = O_wpt['R']

        return _Dv
    
    return func


def _init_wpt( configs, _solver, init_params ):
    t_tof   = init_params[1] / 2 

    r_chs_0_ECI = configs["r_chs_0_ECI"]
    v_chs_0_ECI = configs["v_chs_0_ECI"]

    O_orp, Dv0, Dv1, F = _solver( init_params )

    r_orp_0_ECI = r_chs_0_ECI
    v_orp_0_ECI = v_chs_0_ECI + Dv0

    r_orp_t_ECI, v_orp_t_ECI = UF_FG_S(
        r_orp_0_ECI, v_orp_0_ECI, O_orp, t_tof, MU
    )

    return r_orp_t_ECI, v_orp_t_ECI, O_orp