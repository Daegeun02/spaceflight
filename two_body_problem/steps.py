from optimization import newtonRaphson

from numpy import cos, sin, tan, arctan



## function that find root => estimate Eccentric Anomaly
def func( EccAnm, args ):
    '''
    This function represent the relationship between
    Mean Anomaly and Eccentric Anomaly.

    M = E - e sin( E )
    =>  func( E ) = E - e sin( E ) - M
    '''

    Ecc    = args["Ecc"]
    MeaAnm = args["MeaAnm"]

    return ( EccAnm - Ecc * sin(EccAnm) - MeaAnm )


def grad( EccAnm, args ):
    '''
    This function represent the derivative of the relationship between
    Mean Anomaly and Eccentric Anomaly.

    M = E - e sin( E )
    => grad( E ) = 1 - e cos( E )
    '''

    Ecc = args["Ecc"]

    return ( 1 - Ecc * cos(EccAnm) )


## steps to calculate position and velocity
def step_1(PrP, args, tim):
    '''
    This function estimate Eccenctric Anomaly to calculate True Anomaly.

    From relationship between Mean Anomaly and Eccentric Anomaly,

    M = E - e sin( E ),
    
    find E that make that equation holds with Newton-Raphson Method.
    '''
    
    ## Eccentric Anomaly
    EccAnm = 0
    ## Mean Anomaly
    args["MeaAnm"] = args["AngFrq"] * ( tim - PrP )

    args["EccAnm"] = newtonRaphson( func, grad, EccAnm, args )


def step_2(args):
    '''
    This function calculate True Anomaly by Eccentric Anomaly.

    From relationship between Eccentric Anomaly and True Anomaly,

    tan( nu / 2 ) = sqrt( ( 1 + e ) / ( 1 - e ) ) tan( E / 2 ).
    '''

    EccAnm = args["EccAnm"]
    EccCef = args["EccCef"]

    args["TruAnm"] = arctan( 
        EccCef * tan( EccAnm / 2 )
    )


def step_3(args, position, velocity):
    '''
    This function calculate position and velocity by the relationship 
    with True Anomaly.
    '''

    Ecc    = args["Ecc"]
    TruAnm = args["TruAnm"]
    SemRec = args["SemRec"]
    SemCef = args["SemCef"]

    cTruAnm = cos( TruAnm )
    sTruAnm = sin( TruAnm )

    r = SemRec / ( 1 + Ecc * cTruAnm ) 

    position[0] = r * cTruAnm
    position[1] = r * sTruAnm

    velocity[0] = SemCef * sTruAnm * ( -1 )
    velocity[1] = SemCef * ( Ecc + cTruAnm )

        