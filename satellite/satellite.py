from numpy import zeros, deg2rad



class Satellite:


    def __init__(self):

        self.SemiMajorAxis      = 660
        self.Eccentricity       = 0.7
        self.PerigeePassage     = 0
        self.AscendingNode      = deg2rad(0.0)
        self.OrbitalInclination = deg2rad(0.0)
        self.ArgumentOfPerigee  = deg2rad(0.0)

        self.position = zeros(3)
        self.velocity = zeros(3)