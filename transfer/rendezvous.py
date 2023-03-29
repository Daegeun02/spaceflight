from step01 import step_1
from step02 import step_2
from step03 import step_3



def RENDEZVOUS( r_chs_0_ECI, v_chs_0_ECI, r_trg_0_ECI, v_trg_0_ECI, t_tof, O_chs, O_trg, geometry ):

    r_trg_t_ECI, v_trg_t_ECI = step_1(
        r_trg_0_ECI=r_trg_0_ECI,
        v_trg_0_ECI=v_trg_0_ECI,
        O_trg=O_trg,
        t_tof=t_tof,
        geometry=geometry
    )

    a, o, i, theta_t, theta_0, R = step_2(
        r_chs_0_ECI=r_chs_0_ECI,
        r_trg_t_ECI=r_trg_t_ECI,
        t_tof=t_tof,
        geometry=geometry
    )

    r_chs_0_ORP = R @ r_chs_0_ECI
    v_chs_0_ORP = R @ v_chs_0_ECI

    r_trg_t_ORP = R @ r_trg_t_ECI
    v_trg_t_ORP = R @ v_trg_t_ECI

    Dv0, Dv1, e, w = step_3(
        r_chs_0_ORP=r_chs_0_ORP,
        v_chs_0_ORP=v_chs_0_ORP,
        r_trg_t_ORP=r_trg_t_ORP,
        v_trg_t_ORP=v_trg_t_ORP,
        theta_t=theta_t,
        theta_0=theta_0,
        a=a,
        geometry=geometry
    )

    return Dv0, Dv1, [ a, o, i, e, w ]