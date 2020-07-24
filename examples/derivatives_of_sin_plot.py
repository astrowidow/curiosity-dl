if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from kdnn import Variable
import kdnn.functions as kf

x = Variable(np.linspace(-7, 7, 200), 'x')
y = kf.sin(x)
y.back_prop(create_graph=True)
logs = [y.data.flatten()]

# Calculates third-order differential
for i in range(3):
    logs.append(x.grad.data.flatten())
    gx = x.grad
    x.clear_grad()
    gx.back_prop(create_graph=True)

# Plots the graphs
labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()
