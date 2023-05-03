from numpy.linalg import norm



def optimize( _func, _grad, x0, lr=1e4, itr=500, tol=1e-8 ):

    xK = x0

    fK = _func( x0 )
    gK = _grad( x0 )

    for _ in range( itr ):

        print( list(fK) )

        if ( norm( gK[0] ) < tol ):
            return xK

        xK = xK - lr * gK[0]

        print( xK )

        fP = fK

        fK = _func( xK )

        if ( norm( fK[0] - fP[0] ) < tol ):
            return xK 
        
        if ( norm( fK[0] ) < norm( fP[0] ) ):
            lr *= 1.2
        else:
            lr *= 0.8

        gK = _grad( xK )

    return [fK, gK]