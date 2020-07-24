if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from kdnn import Variable
from kdnn.utils import plot_dot_graph
import kdnn.functions as kf

x = Variable(np.array(1.0))
y = kf.tanh(x)
x.name = 'x'
y.name = 'y'

# first order differentiation
y.back_prop(create_graph=True)

# designates order of differentiation
iters = 7

for i in range(iters):
    gx = x.grad
    x.clear_grad()
    gx.back_prop(create_graph=True)

gx.name = 'gx ' + str(iters)
plot_dot_graph(gx)
