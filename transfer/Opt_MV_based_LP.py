from .AL_func import AL_func, AL_jacb

from numpy import zeros, eye

from numpy.linalg import norm, inv
from numpy.linalg import cond



def solve( args ):

    ## Augmented Lagrangian algorithm
    _xK = zeros(8)          ## Initial point
    uxK = zeros(8)          ## updated point

    _zK = zeros(6)          ## lagrange multiplier
    _uK = 1                 ## augmented variable

    _J = zeros(7)           ## object function
    DJ = zeros((7,8))       ## jacobian matrix

    PJ = zeros(7)           ## previous object
    pf = zeros(1)
    pg = zeros(6)

    func = AL_func( args, _J )
    jacb = AL_jacb( args, DJ )

    tol = 1e-8
    lam = 10 * eye( 8 )
    mit = 1000

    _f, _g = func( _xK )

    for _ in range( mit ):
        ''' Update x with Levenberg-Marquardt algorithm '''
        Df, Dg = jacb( _xK )

        ## optimality condition
        opt_cond = 2 * _f * Df.T + Dg.T @ ( 2 * _uK * _g + _zK )

        if ( norm( opt_cond ) < tol ):
            return _xK

        ## update xK
        dx = inv( DJ.T @ DJ + lam ) @ DJ.T @ _J

        print(cond(DJ.T @ DJ + lam))
        raise ValueError

        uxK[:] = _xK - dx
        ## remember previous objects
        PJ[:] = _J
        pf[:] = _f
        pg[:] = _g
        ## re evaluate objects
        _f, _g = func( uxK )

        ## check tentative iterate
        if ( norm( _J ) < norm( PJ ) ):
            ## update xK
            _xK[:] = uxK
            lam *= 0.8
        else:
            ## do not update xK
            _J[:] = PJ
            _g[:] = pg
            _f    = pf
            lam *= 2.0

        ''' Update z, lagrange multiplier '''
        _zK += 2 * _uK * _g

        ''' Update u, augment variable '''
        if ( norm( _g ) < (0.25) * norm( pg ) ):
            pass
        else:
            _uK *= 2.0

    print('fail to find solution')