from numpy import zeros, deg2rad



class Satellite:


    def __init__(self):

        self.SemiMajorAxis      = 6371 + 660
        self.Eccentricity       = 0.00106
        self.PerigeePassage     = -1
        self.AscendingNode      = deg2rad(90.0)
        self.OrbitalInclination = deg2rad(97.8)
        self.ArgumentOfPerigee  = deg2rad(45.0)

        self.position = zeros(3)
        self.velocity = zeros(3)