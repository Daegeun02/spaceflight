


class NewtonRaphson:


    def __init__(self, tol, itr):

        self.tol = tol
        self.itr = itr


    def solve(self, func, grad, x0, args):
        '''
        root = newtonRaphson(
            func, grad, x0, args
        )

        Finds a root of f(x) = 0 by combining the Newton - Raphson method.
        func: f(x)
        grad: gradient of f(x)
        x0  : initial condition
        args: arguments of func, grad

        *** x is scalar ***
        '''

        tol = self.tol
        itr = self.itr

        if ( func( x0, args ) == 0.0 ):
            return x0

        xK = x0
        dx = 1e8

        for _ in range( itr ):
            f = func( xK, args ) 
            g = grad( xK, args )

            if ( dx < tol ):
                return xK

            if ( g == 0.0 ):
                print('zero division error')
                break

            dx = f / g

            xK -= dx

        print('NewtonRaphson fail to find root of given function')


    def tune_hyper_parameter(self):
        pass