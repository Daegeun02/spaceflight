from two_body_problem import TwoBodyOrbit
from timer            import GlobalTim
from satellite        import Satellite
from geometry         import EARTH
from ground           import GroundControlStation
from transfer         import LambertProblem

from ground           import plot_2_trajectory

from time import time

from numpy import deg2rad



if __name__ == "__main__":

    timer = GlobalTim(
        Hz=1, debug=True
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

    GCS2 = GroundControlStation(
        satellite=Sate1,
        globaltim=timer
    )
    
    timer.start()

    orbit.start()

    GCS1.start()

    Sate1.OrbitalInclination = deg2rad(0.0)
    GCS2.start()

    t1 = time()

    while timer.tim < 5800:

        pass

    t2 = time()

    print( t2 - t1 )

    orbit.estimating = False

    GCS1.tracking = False
    GCS2.tracking = False

    timer.ticking = False

    orbit.join()

    GCS1.join()
    GCS2.join()

    timer.join()

    plot_2_trajectory(
        GCS1.location,
        GCS2.location,
        GCS1.trackIdx,
        GCS2.trackIdx
    )