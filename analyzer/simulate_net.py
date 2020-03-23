from ..python_simulator import CTRNN
import numpy as np
import pickle

# network parameters
NUM_NEURONS = 2
RUN_DURATION = 250
STEP_SIZE = 0.01
TAU_RANGE = (1, 10)
THETA_RANGE = (-15, 15)
W_RANGE = (-15, 15)
G_RANGE = (1, 1)


def run():
    # set up the network
    net = CTRNN.BrainCTRNN(NUM_NEURONS, STEP_SIZE, TAU_RANGE, G_RANGE, THETA_RANGE, W_RANGE)

    # set initial state in specified range
    state_range = [-0.5, 0.5]
    net.randomize_state(state_range)

    # save the agent at the start of the simulation
    fileout_start = open('./agents/net_start_{}nodes'.format(NUM_NEURONS), 'wb')
    pickle.dump(net, fileout_start)
    fileout_start.close()

    # initialize the matrix for saving activation
    num_rows = int(RUN_DURATION / STEP_SIZE)
    output = np.zeros([num_rows, NUM_NEURONS])
    output[0] = net.get_state()
    i_scale = 1 / STEP_SIZE

    # run the simulation for a set number of steps
    for i in np.arange(1, RUN_DURATION*i_scale, STEP_SIZE*i_scale):
        net.euler_step()
        output[int(i)] = net.get_state()

    # save the network output history
    np.save('./agents/net_history_{}nodes'.format(NUM_NEURONS), output)

    # save the agent at the end of the simulation
    fileout_stop = open('./agents/net_stop_{}nodes'.format(NUM_NEURONS), 'wb')
    pickle.dump(net, fileout_stop)
    fileout_start.close()

    return


run()




