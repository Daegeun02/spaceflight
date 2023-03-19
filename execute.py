from two_body_problem import TwoBodyOrbit
from timer            import GlobalTim
from satellite        import Satellite
from geometry         import EARTH
from ground           import GroundControlStation
from transfer         import LambertProblem
from simulator        import Simulator

from time import time, sleep



def do_simulate( dt=10 ):

    timer = GlobalTim()

    earth = EARTH(
        theta_0=0,
        globaltim=timer
    )

    Sate1 = Satellite()

    simul = Simulator(
        satellite=Sate1,
        geometric=earth,
        dt=dt
    )

    simul.start()

    while simul.t < ( Sate1.period ):
        pass

    simul.stop()

    return simul.position


def do_realtime():

    timer = GlobalTim(
        Hz=30
    )

    earth = EARTH(
        theta_0  =0,
        globaltim=timer
    )

    Sate1 = Satellite()

    orbit = TwoBodyOrbit(
        satellite=Sate1,
        geometric=earth,
        globaltim=timer
    )

    GCS1 = GroundControlStation(
        satellite=Sate1,
        globaltim=timer
    )
    
    timer.start()

    orbit.start()

    GCS1.start()

    t1 = time()
    while timer.tim < 30:

        pass
    t2 = time()

    print( t2 - t1 )

    orbit.estimating = False
    GCS1.tracking    = False
    timer.ticking    = False

    orbit.join()
    GCS1.join()
    timer.join()