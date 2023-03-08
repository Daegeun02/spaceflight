from optimization import newtonRaphson

from numpy import cos, sin, tan, arctan



## function that find root => estimate Eccentric Anomaly
def func( EccAnm, args ):

    Ecc    = args["Ecc"]
    MeaAnm = args["MeaAnm"]

    return ( EccAnm - Ecc * sin(EccAnm) - MeaAnm )


def grad( EccAnm, args ):

    Ecc = args["Ecc"]

    return ( 1 - Ecc * cos(EccAnm) )


## steps to calculate position and velocity
def step_1(PrP, args, tim):
    
    ## Eccentric Anomaly
    EccAnm = 0
    ## Mean Anomaly
    args["MeaAnm"] = args["AngFrq"] * ( tim - PrP )

    args["EccAnm"] = newtonRaphson( func, grad, EccAnm, args )


def step_2(args):

    EccAnm = args["EccAnm"]
    EccCef = args["EccCef"]

    args["TruAnm"] = arctan( 
        EccCef * tan( EccAnm / 2 )
    )


def step_3(args, position, velocity):

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

        