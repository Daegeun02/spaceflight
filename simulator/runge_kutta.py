

    
def _RK4(dy_dx, x, y, dt):
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

    k1 = dy_dx( x           , y )
    k2 = dy_dx( x + 0.5 * dt, y + 0.5 * k1 )
    k3 = dy_dx( x + 0.5 * dt, y + 0.5 * k2 )
    k4 = dy_dx( x + dt      , y + k3 )

    y += ( 1.0 / 6.0 ) * ( k1 + 2 * k2 + 2 * k3 + k4 )