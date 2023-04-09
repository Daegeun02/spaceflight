from .step01 import step01
from .step02 import step02
from .step03 import step03



def RENDEZVOUS( r_chs_0_ECI, v_chs_0_ECI, r_trg_0_ECI, v_trg_0_ECI, t_tof, O_chs, O_trg, MU ):

    r_trg_t_ECI, v_trg_t_ECI = step01(
        r_trg_0_ECI=r_trg_0_ECI,
        v_trg_0_ECI=v_trg_0_ECI,
        O_trg=O_trg,
        t_tof=t_tof,
        mu=MU
    )

    print( r_trg_t_ECI, v_trg_t_ECI )

    a, o, i, theta_t, theta_0, R = step02(
        r_chs_0_ECI=r_chs_0_ECI,
        r_trg_t_ECI=r_trg_t_ECI,
        t_tof=t_tof,
        mu=MU
    )

    print( a, o, i, theta_t, theta_0 )

    r_chs_0_ORP = R @ r_chs_0_ECI
    v_chs_0_ORP = R @ v_chs_0_ECI

    r_trg_t_ORP = R @ r_trg_t_ECI
    v_trg_t_ORP = R @ v_trg_t_ECI

    print( r_chs_0_ORP, v_chs_0_ORP )
    print( r_trg_t_ORP, v_trg_t_ORP )

    return 

    Dv0, Dv1, e, w = step03(
        r_chs_0_ORP=r_chs_0_ORP,
        v_chs_0_ORP=v_chs_0_ORP,
        r_trg_t_ORP=r_trg_t_ORP,
        v_trg_t_ORP=v_trg_t_ORP,
        theta_t=theta_t,
        theta_0=theta_0,
        a=8000,
        mu=MU
    )

    return Dv0, Dv1, [ a, o, i, e, w ]