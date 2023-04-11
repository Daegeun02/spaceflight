from transfer import NN_func, NN_jacb

from numpy import zeros



def rendezvous( configs ):

    f = zeros( 10 )
    J = zeros((10,10))

    func = NN_func( configs, f )
    jacb = NN_jacb( configs, J )