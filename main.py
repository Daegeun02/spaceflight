from two_body_problem import TwoBodyOrbit
from timer            import GlobalTim
from satellite        import Satellite
from geometry         import EARTH
from ground           import GroundControlStation
from transfer         import LambertProblem

from time import time



if __name__ == "__main__":

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