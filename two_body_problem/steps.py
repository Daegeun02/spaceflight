from optimization import newtonRaphson

from numpy import cos, sin, tan, arctan



## function that find root => estimate Eccentric Anomaly
def func( E, args ):
    '''
    This function represent the relationship between
    Mean Anomaly and Eccentric Anomaly.

    M = E - e sin( E )
    =>  func( E ) = E - e sin( E ) - M
    '''

    e = args["e"]
    M = args["M"]

    return ( E - e * sin(E) - M )


def grad( E, args ):
    '''
    This function represent the derivative of the relationship between
    Mean Anomaly and Eccentric Anomaly.

    M = E - e sin( E )
    => grad( E ) = 1 - e cos( E )
    '''

    e = args["e"]

    return ( 1 - e * cos(E) )


## steps to calculate position and velocity
def step_1( args, tim ):
    '''
    This function estimate Eccenctric Anomaly to calculate True Anomaly.

    From relationship between Mean Anomaly and Eccentric Anomaly,

    M = E - e sin( E ),
    
    find E that make that equation holds with Newton-Raphson Method.
    '''
    
    ## Eccentric Anomaly
    E = args["E"]
    ## perigee passage
    T = args["T"]

    ## Mean Anomaly
    args["M"] = args["n"] * ( tim - T )

    args["E"] = newtonRaphson( func, grad, E, args )


def step_2(args):
    '''
    This function calculate True Anomaly by Eccentric Anomaly.

    From relationship between Eccentric Anomaly and True Anomaly,

    tan( nu / 2 ) = sqrt( ( 1 + e ) / ( 1 - e ) ) tan( E / 2 ).
    '''

    E  = args["E"]
    eC = args["eC"]

    args["N"] = 2 * arctan( 
        eC * tan( E / 2 )
    )


def step_3(args, position, velocity):
    '''
    This function calculate position and velocity by the relationship 
    with True Anomaly.
    '''

    e  = args["e"]
    N  = args["N"]
    p  = args["p"]
    pC = args["pC"]

    cN = cos( N )
    sN = sin( N )

    r = p / ( 1 + e * cN ) 

    position[0] = r * cN
    position[1] = r * sN

    velocity[0] = pC * sN * ( -1 )
    velocity[1] = pC * ( e + cN )


def step_4(args, position):

    e  = args["e"]
    N  = args["N"]
    p  = args["p"]
    pC = args["pC"]

    cN = cos( N )
    sN = sin( N )

    r = p / ( 1 + e * cN ) 

    position.append([r*cN, r*sN])