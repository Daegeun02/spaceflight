from numpy.linalg import norm



trace = {}

def optimize( _func, _grad, x0, lr=1e4, itr=500, tol=1e-2 ):

    xK = x0

    fK = _func( x0 )[1]
    gK = _grad( x0 )

    for i in range( itr ):

        print( xK[0], ',', xK[1], ',', norm( fK ) )

        if ( norm( gK ) < (tol/lr) ):
            print( xK )
            return xK, trace

        xK = xK - lr * gK

        fP = fK

        fK = _func( xK )[1]

        trace[i] = (xK, fK)

        if ( abs( norm( fK ) - norm( fP ) ) < tol ):
            return xK, trace 
        
        if ( norm( fK ) < norm( fP ) ):
            lr *= 1.2
        else:
            lr *= 0.3

        gK = _grad( xK )

    return [fK, gK]