from geometry import MU

from transfer import RENDEZVOUS

from two_body_problem import estimate



if __name__ == "__main__":

    t_tof = 3000

    O_chs = {
        "a": 7000,
        "o": 0,
        "i": 0,
        "w": 0,
        "e": 0,
        "T": 0
    }

    O_trg = {
        "a": 20000,
        "o": 30,
        "i": 30,
        "w": 30,
        "e": 0.3,
        "T": 0
    }

    r_chs_0_ECI, v_chs_0_ECI = estimate( O_chs, MU, 0 )

    r_trg_0_ECI, v_trg_0_ECI = estimate( O_trg, MU, 0 )

    print(r_chs_0_ECI, v_chs_0_ECI)
    print(r_trg_0_ECI, v_trg_0_ECI)

    solution = RENDEZVOUS(
        r_chs_0_ECI=r_chs_0_ECI,
        v_chs_0_ECI=v_chs_0_ECI,
        r_trg_0_ECI=r_trg_0_ECI,
        v_trg_0_ECI=v_trg_0_ECI,
        t_tof=t_tof,
        O_chs=O_chs,
        O_trg=O_trg,
        MU=MU
    )

    print(solution)