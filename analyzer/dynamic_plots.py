import matplotlib.pyplot as plt
import numpy as np
# import sympy as sm
from ..python_simulator import CTRNN
from scipy.special import expit


NUM_NEURONS = 2
net = CTRNN.BrainCTRNN(2)
net.randomize_state([-0.5, 0.5])
net_history = {'y': np.zeros((10, 2)), 'dy': np.zeros((10, 2))}

for i in range(10):
    net_history['y'][i, :] = net.get_state()
    net_history['dy'][i, :] = net.dy_dt
    net.euler_step()


# Define the sample space (plotting ranges)
ymin = np.amin(net_history['y'])
ymax = np.amax(net_history['y'])

y1 = np.linspace(ymin, ymax, 30)
y2 = np.linspace(ymin, ymax, 30)
Y1, Y2 = np.meshgrid(y1, y2)
dim_y = y1.shape[0]


"""
Define a helper function, which given a point in the state space, 
will tell us what the derivatives of the state elements will be.
One way to do this is to run the model over a single timestep, and extract the derivative information.
"""


def get_derivative(network, state):
    # compute the next state of the network given its current state and the euler equation
    # update the outputs of all neurons
    o = expit(np.multiply(network.G, state + network.Theta))
    # update the state of all neurons
    dy_dt = np.multiply(1 / network.Tau, - state + np.dot(network.W, o) + network.I) * network.step_size
    state += dy_dt
    return tuple(dy_dt)


# vget_derivatives = np.vectorize(get_derivative)


"""
Calculate the state space derivatives across our sample space.
"""

changes_y1 = np.zeros([dim_y, dim_y])
changes_y2 = np.zeros([dim_y, dim_y])

for i in range(dim_y):
    for j in range(dim_y):
        changes = get_derivative(net, np.array([Y1[i,j], Y2[i,j]]))
        changes_y1[i,j] = changes[0]
        changes_y2[i,j] = changes[1]

"""
Plot the phase portrait
We'll use matplotlib quiver function, which wants as arguments the grid of x and y coordinates, and the derivatives of these coordinates.
In the plot we see the locations of stable and unstable equilibria, and can eyeball the trajectories that the system will take through
the state space by following the arrows.
"""

plt.figure(figsize=(10,6))
plt.quiver(Y1, Y2, changes_y1, changes_y2, color='b', alpha=.75)
plt.plot(net_history[:, 0], net_history[:, 1], color='r')
plt.box('off')
plt.xlabel('y1', fontsize=14)
plt.ylabel('y2', fontsize=14)
plt.title('Phase portrait and a single trajectory for 2 neurons randomly initialized', fontsize=16)
plt.show()

# """
# Find steady states
# """
#
# y1var, y2var = sm.symbols('y1, y2')
# dy_dt1 = 1 / net_start.Tau[0] * (-y1var + net_start.W[0,0]*(1/(1+sm.exp(-(y1var+net_start.Theta[0]))))+
#                                  net_start.W[0,1]*(1/(1+sm.exp(-(y2var+net_start.Theta[1])))))
# dy_dt2 = 1 / net_start.Tau[0] * (-y2var + net_start.W[0,0]*(1/(1+sm.exp(-(y1var+net_start.Theta[0]))))+
#                                  net_start.W[1,1]*(1/(1+sm.exp(-(y2var+net_start.Theta[1])))))
#
# # use sympy's way of setting equations to zero
# Y1qual = sm.Eq(dy_dt1, 0)
# Y2qual = sm.Eq(dy_dt2, 0)
#
# # compute fixed points - this is too slow to be usable in this case
# # equilibria = sm.solve((Y1qual, Y2qual), y1var, y2var)
