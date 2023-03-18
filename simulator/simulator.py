from .derivatives import deriv_nu, deriv_r



class Simulator:
    '''
    Simulates satellite's movement.

    With Runge-Kutta 4th order method,
    numerically integrate satellite's state.

    State
    r : distance from the focus of orbit
    nu: True anomaly
    '''


    def __init__(self, satellite, geometric):

        self.satellite = satellite
        self.geometric = geometric


    def start(self):
        pass

    
    def _RK4(self):
        '''
        Runge-Kutta 4th order Method

        y(t+dt) = y(t) + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * dt 

        where 

        k1: f(x(t)           , y(t))
        k2: f(x(t) + (1/2)*dt, y(t) + (1/2)*k1*dt)
        k3: f(x(t) + (1/2)*dt, y(t) + (1/2)*k2*dt)
        k4: f(x(t) + dt      , y(t) + k3*dt)

        f(x, y) = dy/dx
        '''

        

       # Python program to implement Runge Kutta method
# A sample differential equation "dy / dx = (x - y)/2"
def dydx(x, y):
	return ((x - y)/2)

# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def rungeKutta(x0, y0, x, h):
	# Count number of iterations using step size or
	# step height h
	n = (int)((x - x0)/h)
	# Iterate for number of iterations
	y = y0
	for i in range(1, n + 1):
		"Apply Runge Kutta Formulas to find next value of y"
		k1 = h * dydx(x0, y)
		k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
		k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
		k4 = h * dydx(x0 + h, y + k3)

		# Update next value of y
		y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

		# Update next value of x
		x0 = x0 + h
	return y

# Driver method
x0 = 0
y = 1
x = 2
h = 0.2
print ('The value of y at x is:', rungeKutta(x0, y, x, h))

# This code is contributed by Prateek Bhindwar
 