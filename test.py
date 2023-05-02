from geometry import MU

from lambert import LP_solver_without

from numpy import array



_chs_0_ECI = array([5000e3, 0, 5042e3, 0, 7492.18085, 0])
_trg_0_ECI = array([2134089.51, 4094615.131, 5655178.001, -4451.749557, -3952.959997, 4930.183112])
_chs_t_ECI = array([4.36602877e6, 2.43675866e6, 5.04200000e6, 3.17506572e2, 1.77206090e2, 0.00000000e0])
_trg_t_ECI = array([4.36602877e6, 2.43675866e6, 5.04200000e6, 3.17506572e2, 1.77206090e2, 0.00000000e0])

t_tof = 7000



if __name__ == "__main__":

    r_chs_0_ECI = _chs_0_ECI[0:3] / 1000
    v_chs_0_ECI = _chs_0_ECI[3:6] / 1000

    r_trg_t_ECI = _trg_t_ECI[0:3] / 1000
    v_trg_t_ECI = _trg_t_ECI[3:6] / 1000

    O_orp, Dv0, Dv1, F = LP_solver_without( r_chs_0_ECI, v_chs_0_ECI, r_trg_t_ECI, v_trg_t_ECI, t_tof, MU )

    print( O_orp, Dv0, Dv1, F )