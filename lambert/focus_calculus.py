from numpy import sqrt
from numpy import zeros

from numpy.linalg import norm



def get_foci_by_a( a, r_chs_0_ORP, r_trg_t_ORP ):

    r1 = 2*a - norm( r_chs_0_ORP )
    r2 = 2*a - norm( r_trg_t_ORP )

    x = sqrt( ( r1**2 * r2**2 ) / ( r1**2 + r2**2 ) )

    O = ( r_chs_0_ORP + r_trg_t_ORP ) / 2

    D = zeros(3)
    D[0] = r_chs_0_ORP[1] - r_trg_t_ORP[1]
    D[1] = r_trg_t_ORP[0] - r_chs_0_ORP[0]

    F1 = O + D * x
    F2 = O - D * x

    return F1, F2

