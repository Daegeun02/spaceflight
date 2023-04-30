from JAX_universal import UF_func
from JAX_universal import FG_expr

from JAX_optimizer import newtonRaphson



def forward( configs ):

    r0 = "r0"
    v0 = "v0"
    mu = "mu"
    a  = "a"
    t  = "t"

    def _func( t ):

        configs["t"] = t

        _UF_func = UF_func( configs )
        _FG_func = FG_expr( r0, v0, configs )

        x = newtonRaphson( _UF_func, 0.0 )

        _r_v = _FG_func( x, t )

        return x, _r_v
    
    return _func