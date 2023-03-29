## Augmented Lagrangian function
from numpy import cos, sin
from numpy import sqrt
from numpy import eye
from numpy import zeros


'''
args = {
    "mu"     : float,
    "a"      : float,
    "theta_t": float,
    "theta_0": float,

    "r_trg": np.array,
    "v_trg": np.array,
    "r_chs": np.array,
    "v_chs": np.array,
    "r_ch0": np.array,
    "v_ch0": np.array,

    "tri_t": np.array,
    "tri_0": np.array,
    "tri_d": np.array,

    "fandg": np.array
}
'''


def AL_func( args, out ):
    '''
    generate function that evaluate object function

    R^8 -> R^7
    '''
    ## build
    m = args["mu"]
    a = args["a"]
    ## time variant variable
    theta_t = args["theta_t"]
    theta_0 = args["theta_0"]
    ## target final pos, vel
    r_trg = args["r_trg"]
    v_trg = args["v_trg"]
    ## chaser final pos, vel
    r_chs = args["r_chs"]
    v_chs = args["v_chs"]
    ## trianlgular values
    tri_t = args["tri_t"]
    tri_0 = args["tri_0"]
    tri_d = args["tri_d"]

    tri_d[0] = cos( theta_t - theta_0 )
    tri_d[1] = sin( theta_t - theta_0 )

    evl_fg = f_and_g( args )

    def _func( x ):
        '''
        object function

        R^8 -> R^7
        '''
        ## minimizer
        Dv0, Dv1, e, w = x

        ## evaluate parameters
        tri_t[:] = cos( theta_t-w ), sin( theta_t-w )
        tri_0[:] = cos( theta_0-w ), sin( theta_0-w )

        args["p"] = a * ( 1 - e**2 )

        ## calculate
        out[ 0 ] = (0.5) * ( Dv0.T @ Dv0 + Dv1.T @ Dv1 )

        evl_fg( x )

        out[1:4] = r_trg - r_chs
        out[4:7] = v_trg - ( v_chs + Dv1 )

        ##     object  , constraint
        return out[ 0 ], out[1:7]

    return _func


def AL_jacb( args, out ):
    '''
    generate function that evaluate jacobian of object function for x

    R^8 -> R^7 x R^8
    '''
    ## build
    fandg = args["fandg"]

    I = eye(3)

    pgpe = zeros(6)
    pgpw = zeros(6)

    partial_e = partial_fg_e( args, pgpe )
    partial_w = partial_fg_w( args, pgpw )

    def _jacb( x ):
        '''
        jacobian of object function for x

        R^8 -> R^7 x R^8
        '''
        ## minimizer
        Dv0, Dv1, e, w = x

        ## jacobian
        out[ 0 ,0:3] = Dv0
        out[ 0 ,3:6] = Dv1

        out[1:4,0:3] = -fandg[1] * I
        out[4:7,0:3] = -fandg[3] * I

        out[4:7,3:6] = -I

        out[1:7, 7 ] = partial_e( x )
        out[1:7, 8 ] = partial_w( x )

        ##     object      , constraint
        return out[ 0 ,0:8], out[1:7,0:8]

    return _jacb


def f_and_g( args ):
    ## build
    m = args["mu"]
    a = args["a"]
    ## chaser final pos, vel
    r_chs = args["r_chs"]
    v_chs = args["v_chs"]
    ## chaser initial pos, vel
    r_ch0 = args["r_ch0"]
    v_ch0 = args["v_ch0"]

    ## triangular values
    tri_t = args["tri_t"]
    tri_0 = args["tri_0"]
    tri_d = args["tri_d"]

    fandg = args["fandg"]

    ct0, st0 = tri_d

    def _f_and_g( x ):
        ## minimizer
        Dv0, Dv1, e, w = x

        ## evaluate parameters
        ct, st = tri_t
        c0, s0 = tri_0

        p = args["p"]

        ## calculate position
        f = ( e * ct + ct0 ) / ( 1 + e * ct )
        g = -sqrt( p**3 / m ) * ct0 / ( ( 1 + e * ct ) * ( 1 + e * c0 ) )

        r_chs[:] = f * r_ch0 + g * ( v_ch0 + Dv0 )

        ## calculate velocity
        fdot = -sqrt( m / p**3 ) * ( e * s0 + e * st + st0 )
        gdot = -( e * s0 + st0 )

        v_chs[:] = fdot * r_ch0 + gdot * ( v_ch0 + Dv0 )

        fandg[:] = f, g, fdot, gdot
        ## end

    return _f_and_g


def partial_fg_e( args, out ):
    ## build
    m = args["mu"]
    a = args["a"]
    ## chaser initial pos, vel
    r_ch0 = args["r_ch0"]
    v_ch0 = args["v_ch0"]
    ## tri theta_t
    tri_t = args["tri_t"]
    tri_0 = args["tri_0"]
    tri_d = args["tri_d"]

    ct0, st0 = tri_d

    def _partial_fg_e( x ):
        ## minimizer
        Dv0, Dv1, e, w = x

        ## evaluate parameters
        ct, st = tri_t
        c0, s0 = tri_0
        
        p = args["p"]

        cte = 1 + e * ct
        c0e = 1 + e * c0

        E = ( 1 - e**2 )

        ## calculate partial position for e
        p_fpe = ct * ( 1 - ct0 ) / ( 1 + e * ct )**2

        p_gpe = 0
        p_gpe += (1.5) * sqrt( a * p / m ) * ( 2 * a * e ) * cte * ct0
        p_gpe += sqrt( p**3 / m ) * ( ct * c0e + cte * c0 )
        p_gpe *= ct0 / ( ( cte**2 ) * ( c0e**2 ) )

        out[0:3] = (-1) * ( p_fpe * r_ch0 + p_gpe * ( v_ch0 + Dv0 ) )

        ## calculate partial velocity for e
        pdfpe = 0
        pdfpe -= ( s0 + st ) * E**3
        pdfpe -= ( e * s0 + e * st + st0 ) * (1.5) * E * 2 * e
        pdfpe *= sqrt( m / a**3 ) / ( E**6 )

        pdgpe = -s0

        out[3:6] = (-1) * ( pdfpe * r_ch0 + pdgpe * ( v_ch0 + Dv0 ) )

        return out

    return _partial_fg_e


def partial_fg_w( args, out ):
    ## build
    m = args["mu"]
    a = args["a"]
    ## chaser initial pos, vel
    r_ch0 = args["r_ch0"]
    v_ch0 = args["v_ch0"]
    ## tri theta_t
    tri_t = args["tri_t"]
    tri_0 = args["tri_0"]
    tri_d = args["tri_d"]

    ct0, st0 = tri_d

    def _partial_fg_w( x ):
        ## minimizer
        Dv0, Dv1, e, w = x

        ## evaluate parameters
        ct, st = tri_t
        c0, s0 = tri_0
        
        p = args["p"]

        cte = 1 + e * ct
        c0e = 1 + e * c0

        ## calculate partial position for w
        p_fpw = 0
        p_fpw += e * st * ( 1 - ct0 )
        p_fpw /= ( cte**2 )

        p_gpw = 0
        p_gpw += ( e * st * c0e + e * s0 * cte )
        p_gpw *= ( cte**2 * c0e**2 )
        p_gpw *= ( -sqrt( p**3 / m ) * ct0 )

        out[0:3] = (-1) * ( p_fpw * r_ch0 + p_gpw * ( v_ch0 + Dv0 ) )

        ## calculate partial velocity for w
        pdfpw = 0
        pdfpw += ( ct + c0 )
        pdfpw *= ( e * sqrt( m / p**3 ) )

        pdgpw = e * c0

        out[3:6] = (-1) * ( pdfpw * r_ch0 + pdgpw * ( v_ch0 + Dv0 ) )

        return out

    return _partial_fg_w