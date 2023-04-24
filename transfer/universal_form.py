from numpy import cos, sin
from numpy import sqrt
from numpy import dot

from numpy.linalg import norm



def UF_func( configs ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    sqrt_m = sqrt( m )

    _r0 = norm( r0 )

    def _func( x, t=t, a=a, v0=v0 ):
        '''
        This function calculates energy equation
        for universal variable.

        <parameter>
        x : universal variable
        t : time
        a : semimajor axis
        v0: initial velocity
        '''
        sqrt_a = sqrt( a )

        out = 0.0
        out += a * ( x - sqrt_a * sin( x / sqrt_a ) )
        out += ( dot( r0, v0 ) / sqrt_m ) * a * ( 1 - cos( x / sqrt_a ) )
        out += _r0 * sqrt_a * sin( x / sqrt_a )
        out -= sqrt_m * t

        return out

    return _func


def UF_grad( configs ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    sqrt_m = sqrt( m )

    _r0 = norm( r0 )

    def _grad( x, t=t, a=a, v0=v0 ):
    
        sqrt_a = sqrt( a )

        out = 0.0
        out += a * ( 1 - cos( x / sqrt_a ) )
        out += ( dot( r0, v0 ) / sqrt_m ) * sqrt_a * sin( x / sqrt_a )
        out += _r0 * cos( x / sqrt_a )

        return out

    return _grad


def f_and_g_expression( x, r0, v0, configs ):

    m = configs["mu"]
    a = configs["a"]
    t = configs["t"]

    sqrt_a = sqrt( a )
    sqrt_m = sqrt( m )
    _r0 = norm( r0 )
    _v0 = norm( v0 )

    f = 1 - ( a / _r0 ) * ( 1 - cos( x / sqrt_a ) )
    g = t - ( a / sqrt_m ) * ( x - sqrt_a * sin( x / sqrt_a ) )

    r = f * r0 + g * v0

    _r = norm( r )

    fdot = (-1) * ( ( sqrt_a * sqrt_m ) / ( _r0 * _r ) ) * sin( x / sqrt_a )
    gdot = 1 - ( a / _r ) * ( 1 - cos( x / sqrt_a ) )

    v = fdot * r0 + gdot * v0

    return r, v 


def FG_func( configs, out ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    _r0 = norm( r0 )

    sqrt_m = sqrt( m )

    def _func( x, t=t, a=a, v0=v0 ):
        '''
        This function estimates state at time t
        based on f and g expression.

        <parameter>
        x : universal variable
        t : time
        a : semimajor axis
        v0: initial velocity
        '''
        sqrt_a = sqrt( a )

        f = 1 - ( a / _r0 ) * ( 1 - cos( x / sqrt_a ) )
        g = t - ( a / sqrt_m ) * ( x - sqrt_a * sin( x / sqrt_a ) )

        out[0:3] = f * r0 + g * v0

        _r = norm( out[0:3] )

        fdot = (-1) * ( ( sqrt_a * sqrt_m ) / ( _r0 * _r ) ) * sin( x / sqrt_a )
        gdot = 1 - ( a / _r ) * ( 1 - cos( x / sqrt_a ) )

        out[3:6] = fdot * r0 + gdot * v0

        return out
    
    return _func