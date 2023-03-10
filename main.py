from two_body_problem import TwoBodyOrbit
from timer            import GlobalTim
from satellite        import Satellite
from geometry         import EARTH
from ground           import GroundControlStation

from time import time



if __name__ == "__main__":

    timer = GlobalTim(Hz=20)

    earth = EARTH()

    Sate1 = Satellite()

    orbit = TwoBodyOrbit(
        satellite=Sate1,
        geometric=earth,
        globaltim=timer
    )

    GCS = GroundControlStation(
        satellite=Sate1,
        globaltim=timer
    )

    
    timer.start()

    orbit.start()

    GCS.start()

    t1 = time()

    while timer.tim < 3:

        pass

    t2 = time()

    print( t2 - t1 )

    orbit.estimating = False

    orbit.join()

    GCS.tracking = False

    GCS.join()

    timer.ticking = False

    timer.join()