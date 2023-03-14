


GRAVCONST = 6.674e-11

EARTHMASS = 5.9742e24

class EARTH:


    def __init__(self):

        self.mu = GRAVCONST * EARTHMASS  / (1000 ** 3)