


GRAVCONST = 6.674e-11

EARTHMASS = 5.9742e24

class EARTH:


    def __init__(self):

        self.mu = GRAVCONST * EARTHMASS  / (1000 ** 3)

        self.rotate_velocity = 464 / 1000 ## km/s

        self.rotating = True


    def rotate(self):
        pass