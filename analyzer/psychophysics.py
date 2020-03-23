"""
Which analysis will be performed?
1. Plot agent behavior.
1. Neural activation of all 8 neurons throughout the trial.
2. Activation for different types of neurons might need to be plotted separately
    due to the differences in activation range.
3. Activation of neurons compared to neural input they are receiving.
4. Activation of sensory and motor neurons compared to the target-tracker distance.
7. Plot evolved network weights (Hinton diagram).
5. Lesion of auditory information (from the start of the trial vs at specific points).
6. Lesion of visual information (from the start of the trial vs at specific points).
8. Dynamical plots for activation vectors given a particular input.
9. Correlation (mutual information? coupling?) between neural activations of two agents.
10. Prediction between neural activation of one agent and the output of the other.
11. Compare behavioral patterns with the psychological experiment results.

- adding noise
- network lesioning
- pca on neuron activations - see if top components look similar across agents
- cross-correlations
- EEG analysis toolbox in python?
- cross-recurrence
- mutual information
- correlation over all trials or at decision points?
- check generalization again, show that it works
- weight matrix correlation with activation correlation?

sync measures:
imaginary coherence
projected power correlations
for each frequency


seeded
392678
513311

random
176176
392678*
424698
628329
717077
812711
914463*


"""
from . import analyze as az
import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
from ..python_simulator import simulate
import pickle
from scipy import signal
from copy import deepcopy
from scipy.special import expit
# import sympy as sm


agent_directory = "Agents/joint/direct/random/914463"
#agent_directory = "Agents/joint/direct/seeded/628329"
gen_files = fnmatch.filter(os.listdir(agent_directory), 'gen*')
gen_numbers = [int(x[3:]) for x in gen_files]
last_gen = max(gen_numbers)
config = az.load_config(agent_directory)

# Plot fitness
# az.plot_fitness(agent_directory)

td, ag1, ag2 = az.run_best_pair(agent_directory, last_gen)


agent_directory_176 = "Agents/joint/direct/random/176176"
gen_files_176 = fnmatch.filter(os.listdir(agent_directory_176), 'gen*')
gen_numbers_176 = [int(x[3:]) for x in gen_files_176]
last_gen_176 = max(gen_numbers_176)
config_176 = az.load_config(agent_directory_176)
td_176, ag1_176, ag2_176 = az.run_best_pair(agent_directory_176, last_gen_176)


fig = plt.figure(figsize=(8, 4))

ax = fig.add_subplot(1, 2, 1)
ax.plot(td['target_pos'][1], label='Target position')
ax.plot(td['tracker_pos'][1], label='Tracker position')
ax.plot(td['tracker_v'][1], label='Tracker velocity')
ax.plot(td['keypress'][1][:, 0], label='Left motor')
ax.plot(td['keypress'][1][:, 1], label='Right motor')
ax.set_title("Joint strategy")

ax = fig.add_subplot(1, 2, 2)
ax.plot(td_176['target_pos'][1], label='x target')
ax.plot(td_176['tracker_pos'][1], label='x tracker')
ax.plot(td_176['tracker_v'][1], label='v tracker')
ax.plot(td_176['keypress'][1][:, 0], label='v left')
ax.plot(td_176['keypress'][1][:, 1], label='v right')
ax.set_title("Independent strategy")
ax.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1)

plt.tight_layout()
plt.savefig('strategies.eps')


output = open('td_914463.pkl', 'wb')
pickle.dump(td, output)
output.close()

resampled_td = az.resample_trials(td)

output = open('resampled_td_914463.pkl', 'wb')
pickle.dump(resampled_td, output)
output.close()

td_left = az.run_agent_from_best_pair(agent_directory, last_gen, 'left')
td_right = az.run_agent_from_best_pair(agent_directory, last_gen, 'right')

lims = [config['evaluation_params']['screen_width'][0]-1, config['evaluation_params']['screen_width'][1]+1]
az.plot_data(td_left, "all", "Left agent alone", lims)
az.plot_data(td_right, "all", "Right agent alone", lims)


def plot_data(trial_data, trial_num, to_plot, fig_title):
    if trial_num == "all":
        num_trials = len(trial_data['target_pos'])
        num_cols = num_trials/2

        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(fig_title)

        for p in range(num_trials):
            ax = fig.add_subplot(2, num_cols, p + 1)
            if to_plot == "behavior":
                ax.plot(trial_data['target_pos'][p], label='Target position')
                ax.plot(trial_data['tracker_pos'][p], label='Tracker position')
                ax.plot(trial_data['tracker_v'][p], label='Tracker velocity')
                ax.plot(trial_data['keypress'][p][:, 0], label='Left motor')
                ax.plot(trial_data['keypress'][p][:, 1], label='Right motor')
            elif to_plot == "activation_all":
                ax.plot(trial_data['brain_state_a1'][p][:, 0], label='n1')
                ax.plot(trial_data['brain_state_a1'][p][:, 1], label='n2')
                ax.plot(trial_data['brain_state_a1'][p][:, 2], label='n3')
                ax.plot(trial_data['brain_state_a1'][p][:, 3], label='n4')
                ax.plot(trial_data['brain_state_a1'][p][:, 4], label='n5')
                ax.plot(trial_data['brain_state_a1'][p][:, 5], label='n6')
                ax.plot(trial_data['brain_state_a1'][p][:, 6], label='n7')
                ax.plot(trial_data['brain_state_a1'][p][:, 7], label='n8')
            elif to_plot == "input_all":
                ax.plot(trial_data['input_a1'][p][:, 0], label='n1')
                ax.plot(trial_data['input_a1'][p][:, 1], label='n2')
                ax.plot(trial_data['input_a1'][p][:, 2], label='n3')
                ax.plot(trial_data['input_a1'][p][:, 3], label='n4')
                ax.plot(trial_data['input_a1'][p][:, 4], label='n5')
                ax.plot(trial_data['input_a1'][p][:, 5], label='n6')
                ax.plot(trial_data['input_a1'][p][:, 6], label='n7')
                ax.plot(trial_data['input_a1'][p][:, 7], label='n8')
            elif to_plot == "output_all":
                ax.plot(trial_data['output_a1'][p][:, 0], label='n7_a1')
                ax.plot(trial_data['output_a1'][p][:, 1], label='n8_a1', ls='--')
                ax.plot(trial_data['output_a2'][p][:, 0], label='n7_a2', ls='--')
                ax.plot(trial_data['output_a2'][p][:, 1], label='n8_a2')

    else:
        if to_plot == "behavior":
            plt.plot(trial_data['target_pos'][trial_num], label='Target position')
            plt.plot(trial_data['tracker_pos'][trial_num], label='Tracker position')
            plt.plot(trial_data['tracker_v'][trial_num], label='Tracker velocity')
            plt.plot(trial_data['keypress'][trial_num][:, 0], label='Left motor')
            plt.plot(trial_data['keypress'][trial_num][:, 1], label='Right motor')
        elif to_plot == "activation_all":
            plt.plot(trial_data['brain_state'][trial_num][:, 0], label='n1')
            plt.plot(trial_data['brain_state'][trial_num][:, 1], label='n2')
            plt.plot(trial_data['brain_state'][trial_num][:, 2], label='n3')
            plt.plot(trial_data['brain_state'][trial_num][:, 3], label='n4')
            plt.plot(trial_data['brain_state'][trial_num][:, 4], label='n5')
            plt.plot(trial_data['brain_state'][trial_num][:, 5], label='n6')
            plt.plot(trial_data['brain_state'][trial_num][:, 6], label='n7')
            plt.plot(trial_data['brain_state'][trial_num][:, 7], label='n8')
        elif to_plot == "input_all":
            plt.plot(trial_data['input'][trial_num][:, 0], label='n1')
            plt.plot(trial_data['input'][trial_num][:, 1], label='n2')
            plt.plot(trial_data['input'][trial_num][:, 2], label='n3')
            plt.plot(trial_data['input'][trial_num][:, 3], label='n4')
            plt.plot(trial_data['input'][trial_num][:, 4], label='n5')
            plt.plot(trial_data['input'][trial_num][:, 5], label='n6')
            plt.plot(trial_data['input'][trial_num][:, 6], label='n7')
            plt.plot(trial_data['input'][trial_num][:, 7], label='n8')
        elif to_plot == "output_all":
            plt.plot(trial_data['output'][trial_num][:, 0], label='n7')
            plt.plot(trial_data['output'][trial_num][:, 1], label='n8')

    plt.legend()
    plt.show()


plot_data(td, "all", "behavior", "Trial behavior")
plot_data(td, "all", "activation_all", "Neuronal activation")
plot_data(td, 0, "input_all", "Input to neurons")
plot_data(td, "all", "output_all", "Neuronal output")


"""
Generalization:
- other target and tracker speeds
- other initial target location
- other boundary size
- plot performance score for these tests
- target reversing before the border
"""

td_gen = az.check_joint_generalization(agent_directory, ag1, ag2)
# plot_data(td_gen['impact'], "all", "behavior", "Trial behavior")
plot_data(td_gen['startpos'], "all", "behavior", "Trial behavior")
plot_data(td_gen['speed'], "all", "behavior", "Trial behavior")
plot_data(td_gen['turns'], "all", "behavior", "Trial behavior")
plot_data(td_gen['width'], "all", "behavior", "Trial behavior")

meanf = (np.mean(td_gen['startpos']['fitness']) + np.mean(td_gen['speed']['fitness']) +
         np.mean(td_gen['turns']['fitness']) + np.mean(td_gen['width']['fitness'])) / 4



plt.plot(td['target_pos'][4], label='Target position')
plt.plot(td['tracker_pos'][4], label='Tracker position')

plt.plot(td_gen['width']['target_pos'][4], label='Target position')
plt.plot(td_gen['width']['tracker_pos'][4], label='Tracker position')



td_left = az.run_agent_from_best_pair(agent_directory, last_gen, 'left')
td_right = az.run_agent_from_best_pair(agent_directory, last_gen, 'right')

lims = [config['evaluation_params']['screen_width'][0]-1, config['evaluation_params']['screen_width'][1]+1]
az.plot_data(td_left, "all", "Left agent alone", lims)
az.plot_data(td_right, "all", "Right agent alone", lims)


def plot_corepresentation(trial_data, trial_num, neuron_num):
    plt.plot(trial_data['brain_state_a1'][trial_num][:, neuron_num],
             trial_data['output_a2'][trial_num][:, 1],'.')


plot_corepresentation(td, 0, 0)
plot_corepresentation(td, 0, 1)
plot_corepresentation(td, 0, 2)
plot_corepresentation(td, 0, 3)
plot_corepresentation(td, 0, 4)
plot_corepresentation(td, 0, 5)
plot_corepresentation(td, 0, 6)
plot_corepresentation(td, 0, 7)


def plot_inputs(agent, trial_data, neuron, title, agent_num):
    input_key = 'input_a' + str(agent_num)
    activation_key = 'brain_state_a' + str(agent_num)
    output_key = 'output_a' + str(agent_num)

    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1)
        if neuron in range(1, 5):
            x = trial_data[input_key][p][:, neuron-1]/agent.VW[neuron-1]
        else:
            x = trial_data[input_key][p][:, neuron - 1] / agent.AW[neuron - 5]
        y = trial_data[activation_key][p][:, neuron-1]
        # y = trial_data[output_key][p][:, neuron - 1]
        # ax.set_xlabel("Input")
        # ax.set_ylabel("Activation")
        # ax.set_ylabel("Output")
        ax.plot(x)
        ax.plot(y)
        # ax.plot(x[0], y[0], 'ro', markersize=10)
        # ax.plot(x[-1], y[-1], 'ro', markersize=10, mfc='none')


plot_inputs(ag1, td, 1, "Distance to left border: input vs activation", 1)
plot_inputs(ag1, td, 2, "Distance to right border: input vs activation", 1)
plot_inputs(ag1, td, 3, "Left eye distance to target: input vs activation", 1)
plot_inputs(ag1, td, 4, "Right eye distance to target: input vs activation", 1)
plot_inputs(ag1, td, 5, "Left ear: input vs activation", 1)
plot_inputs(ag1, td, 6, "Right ear: input vs activation", 1)

plot_inputs(ag2, td, 1, "Distance to left border: input vs activation", 2)
plot_inputs(ag2, td, 2, "Distance to right border: input vs activation", 2)
plot_inputs(ag2, td, 3, "Left eye distance to target: input vs activation", 2)
plot_inputs(ag2, td, 4, "Right eye distance to target: input vs activation", 2)
plot_inputs(ag2, td, 5, "Left ear: input vs activation", 2)
plot_inputs(ag2, td, 6, "Right ear: input vs activation", 2)


def plot_outputs_a1(trial_data, title):
    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1)
        x = trial_data['output_a1'][p][:, 6]
        y = trial_data['output_a1'][p][:, 7]
        z = trial_data['keypress'][p][:, 0]
        ax.plot(x, label="n7")
        ax.plot(y, label="n8")
        ax.plot(z, label="left motor")
    plt.legend()


def plot_outputs_a2(trial_data, title):
    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1, projection='3d')
        x = trial_data['output_a2'][p][:, 6]
        y = trial_data['output_a2'][p][:, 7]
        z = trial_data['keypress'][p][:, 1]
        ax.set_xlabel("Output of n7")
        ax.set_ylabel("Output of n8")
        ax.set_zlabel("Motor activation")
        ax.plot(x, y, z)


fig = plt.figure(figsize=(8, 4))
fig.suptitle("Motor neurons output vs motor activation")
plt.plot(td['output_a1'][1][:, 6], label="ag1 n7")
plt.plot(td['output_a1'][1][:, 7], label="ag1 n8")
scaled_left = td['keypress'][1][:, 0] / 10
plt.plot(scaled_left, label="left motor")
plt.plot(td['output_a2'][1][:, 6], label="ag2 n7")
plt.plot(td['output_a2'][1][:, 7], label="ag2 n8")
scaled_right = td['keypress'][1][:, 1] / 10
plt.plot(scaled_right, label="right motor")
plt.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1, framealpha=1)
fig.savefig("motors.eps")

fig = plt.figure(figsize=(8, 4))
fig.suptitle("Motor neurons output vs motor activation")
plt.plot(td['brain_state_a1'][1][:, 6], label="ag1 n7")
plt.plot(td['brain_state_a1'][1][:, 7], label="ag1 n8")
scaled_left = td['keypress'][1][:, 0] / 10
plt.plot(scaled_left, label="left motor")
plt.plot(td['brain_state_a2'][1][:, 6], label="ag2 n7")
plt.plot(td['brain_state_a2'][1][:, 7], label="ag2 n8")
scaled_right = td['keypress'][1][:, 1] / 10
plt.plot(scaled_right, label="right motor")
plt.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1, framealpha=1)
fig.savefig("motors.eps")


plot_outputs_a1(td, "Left agent motor neuron outputs vs left motor activation")
plot_outputs_a2(td, "Right agent motor neuron outputs vs right motor activation")


def plot_input_output(agent, agent_num, trial_data, trial_num, title):
    input_key = 'input_a' + str(agent_num)
    if trial_num == "all":
        num_trials = len(trial_data['target_pos'])
        num_cols = num_trials / 2
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title)

        for p in range(num_trials):
            ax = fig.add_subplot(2, num_cols, p + 1)
            d = td['target_pos'][p] - td['tracker_pos'][p]
            i1 = trial_data[input_key][p][:, 2]/agent.VW[2]
            i2 = trial_data[input_key][p][:, 3]/agent.VW[3]
            o1 = trial_data['keypress'][p][:, 0]
            o2 = trial_data['keypress'][p][:, 1]
            ax.plot(d, label="distance to target")
            ax.plot(i1, label="distance to left eye")
            ax.plot(i2, label="distance to right eye")
            ax.plot(o1, label="left motor activation")
            ax.plot(o2, label="right motor activation")
        ax.legend(bbox_to_anchor=(0., 0.), ncol=2, loc=2, borderaxespad=0.)
    else:
        d = td['target_pos'][trial_num] - td['tracker_pos'][trial_num]
        o1 = trial_data['keypress'][trial_num][:, 0]
        o2 = trial_data['keypress'][trial_num][:, 1]
        plt.plot(d, label="distance to target")
        plt.plot(o1, label="left motor activation")
        plt.plot(o2, label="right motor activation")
        plt.legend(bbox_to_anchor=(0., 0.), ncol=2, loc=2, borderaxespad=0.)


# this is the same for both agents
plot_input_output(ag1, 1, td, 'all', "Distance to target vs motor activation")

# Object centered motion over time plot: distance to target over time
plt.plot(td['target_pos'][0] - td['tracker_pos'][0])


# the agent's motion over time in response to the target held constantly at different distances in different locations
td_immobile = az.run_best_pair_simple(agent_directory, ag1, ag2)

plot_data(td_immobile, "all", "behavior", "Trial behavior")
plot_data(td_immobile, "all", "activation_all", "Neuronal activation")
plot_data(td_immobile, "all", "input_all", "Input to neurons")
plot_data(td_immobile, "all", "output_all", "Neuronal output")


def linmap(vin, rin, rout):
    """
    Map a vector between 2 ranges.
    :param vin: input vector to be mapped
    :param rin: range of vin to map from
    :param rout: range to map to
    :return: mapped output vector
    :rtype np.ndarray
    """
    a = rin[0]
    b = rin[1]
    c = rout[0]
    d = rout[1]
    return ((c + d) + (d - c) * ((2 * vin - (a + b)) / (b - a))) / 2


def get_params_from_genotype(agent):
    """Take a genotype vector and read off agent parameters."""
    genorest, vw, aw, mw = np.hsplit(agent.genotype, agent.crossover_points[-3:])
    unflattened = genorest.reshape(agent.n_evp + agent.brain.N, agent.brain.N, order='F')
    tau, theta, w = (np.squeeze(a) for a in np.vsplit(unflattened, [1, 2]))
    Tau = linmap(tau, agent.gene_range, config["network_params"]["tau_range"])
    Theta = linmap(theta, agent.gene_range, config["network_params"]["theta_range"])
    W = linmap(w, agent.gene_range, config["network_params"]["w_range"]).transpose()
    return Tau, Theta, W


tau1, theta1, w1 = get_params_from_genotype(ag1)
tau2, theta2, w2 = get_params_from_genotype(ag2)


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix.
    Positive and negative values are represented by white and black squares, respectively,
    and the size of each square represents the magnitude of each value.
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    if matrix.ndim == 2:
        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)
    else:
        for (x), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, 1], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


hinton(w1, max_weight=15)
hinton(tau1, max_weight=100)
hinton(theta1, max_weight=15)
hinton(ag1.VW, max_weight=100)
hinton(ag1.AW, max_weight=100)
hinton(ag1.MW, max_weight=10)


hinton(w2, max_weight=15)
hinton(tau2, max_weight=100)
hinton(theta2, max_weight=15)
hinton(ag2.VW, max_weight=100)
hinton(ag2.AW, max_weight=100)
hinton(ag2.MW, max_weight=10)


"""Dynamic plots"""

"""
Define a helper function, which given a point in the state space, will tell us what the derivatives of
the state elements will be. One way to do this is to run the model over a single timestep, and extract
the derivative information.
def expit(x):
    return 1 / (1 + np.exp(-x))
def get_derivative(network, state):
    # compute the next state of the network given its current state and the euler equation
    # update the outputs of all neurons
    o = expit(np.multiply(network.G, state + network.Theta))
    # update the state of all neurons
    dy_dt = np.multiply(1 / network.Tau, - state + np.dot(network.W, o) + network.I) * network.h
    state += dy_dt
    return tuple(dy_dt)
# vget_derivatives = np.vectorize(get_derivative)
"""


def plot_motor_portrait(trial_data, trial_num):
    # Load the activation history of two motor neurons
    # net_history = trial_data['brain_state'][trial_num][:, 6:8]
    net_history = np.stack((td['brain_state_a1'][0][:, 6], td['brain_state_a2'][0][:, 7]), axis=-1)

    # Define the sample space (plotting ranges)
    ymin = np.amin(net_history)
    ymax = np.amax(net_history)
    # Define plotting grid
    y1 = np.linspace(ymin, ymax, 30)
    y2 = np.linspace(ymin, ymax, 30)
    Y1, Y2 = np.meshgrid(y1, y2)
    # Load the derivatives
    changes_y1 = trial_data['derivatives_a1'][trial_num][:, 6]
    changes_y2 = trial_data['derivatives_a2'][trial_num][:, 7]

    # Plot the phase portrait
    # quiver function takes the grid of x-y coordinates and their derivatives
    plt.figure(figsize=(10, 6))
    plt.quiver(Y1, Y2, changes_y1, changes_y2, color='b', alpha=.75)
    plt.plot(net_history[:, 0], net_history[:, 1], color='r')
    plt.box('off')
    plt.xlabel('y1', fontsize=14)
    plt.ylabel('y2', fontsize=14)
    plt.title('Phase portrait of two motor neurons', fontsize=16)
    plt.show()


def plot_velocity_portrait():
    positions = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20]
    # Define the sample space (plotting ranges)
    ymin = -20
    ymax = 20
    # Define plotting grid
    y1 = np.linspace(ymin, ymax, 11)
    y2 = np.linspace(ymin, ymax, 11)
    Y1, Y2 = np.meshgrid(y1, y2)

    changes_y1 = np.zeros((11, 11))
    # Load the derivatives
    for pos1 in positions:
        for pos2 in positions:
            changes_y1[pos1, pos2] = td_steady[str(pos1)]['tracker_v'][pos2][-1]

    changes_y2 = np.zeros((11, 11))

    # Plot the phase portrait
    # quiver function takes the grid of x-y coordinates and their derivatives
    plt.figure(figsize=(10, 6))
    plt.quiver(Y1, Y2, changes_y1, changes_y2, color='b', alpha=.75)
    plt.box('off')
    plt.xlabel('target position', fontsize=14)
    plt.ylabel('starting tracker position', fontsize=14)
    plt.show()


"""
We will plot the phase portraits of 2 motor neurons in different situations:
- constant input from the target position when the tracker is in the middle at the start
- constant input from the target position when the tracker is midway towards the border (approaching)
- constant input from the target position when the tracker is immediately in front of the border
- constant input from the target position when the tracker is right at the border
- constant input from the target position when the tracker is moving away from the border
"""

# constant target input with tracker in the center
plot_motor_portrait(td_immobile, 0)
plot_motor_portrait(td_immobile, 3)



"""
Lesion studies:
- no auditory information
- no visual information of the border
- no visual information of the tracker
- loss of perceptual information at critical points
"""

lesion_test = simulate.LesionedSimulation(config['network_params']['step_size'], config['evaluation_params'])
td_deaf_start = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "start", "auditory", savedata=True)
td_deaf_before_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "before_midturn",
                                                   "auditory", savedata=True)
td_deaf_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "midturn", "auditory", savedata=True)

plot_data(td_deaf_start, "all", "behavior", "Trial behavior")
plot_data(td_deaf_before_half, "all", "behavior", "Trial behavior")
plot_data(td_deaf_half, "all", "behavior", "Trial behavior")


td_borderblind_start = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "start",
                                                    "visual_border", savedata=True)
td_borderblind_before_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "before_midturn",
                                                          "visual_border", savedata=True)
td_borderblind_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "midturn",
                                                   "visual_border", savedata=True)
plot_data(td_borderblind_start, "all", "behavior", "Trial behavior")
plot_data(td_borderblind_before_half, "all", "behavior", "Trial behavior")
plot_data(td_borderblind_half, "all", "behavior", "Trial behavior")


td_targetblind_start = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "start",
                                                    "visual_target", savedata=True)
td_targetblind_before_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "before_midturn",
                                                          "visual_target", savedata=True)
td_targetblind_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "midturn",
                                                   "visual_target", savedata=True)
plot_data(td_targetblind_start, "all", "behavior", "Trial behavior")
plot_data(td_targetblind_before_half, "all", "behavior", "Trial behavior")
plot_data(td_targetblind_half, "all", "behavior", "Trial behavior")


fig = plt.figure(figsize=(7, 5))

ax = fig.add_subplot(3, 2, 1)
ax.plot(td_gen['speed']['target_pos'][0], label='Target position')
ax.plot(td_gen['speed']['tracker_pos'][0], label='Tracker position')
ax.set_title("Faster target")

ax = fig.add_subplot(3, 2, 2)
ax.plot(td_deaf_start['target_pos'][0], label='Target position')
ax.plot(td_deaf_start['tracker_pos'][0], label='Tracker position')
ax.set_title("Deaf agents")

ax = fig.add_subplot(3, 2, 3)
ax.plot(td_gen['speed']['target_pos'][1], label='x target')
ax.plot(td_gen['speed']['tracker_pos'][1], label='x tracker')
ax.set_title("Slower target")

ax = fig.add_subplot(3, 2, 4)
ax.plot(td_borderblind_start['target_pos'][0], label='x target')
ax.plot(td_borderblind_start['tracker_pos'][0], label='x tracker')
ax.set_title("Border-blind agents")

ax = fig.add_subplot(3, 2, 5)
ax.plot(td_gen['width']['target_pos'][4], label='x target')
ax.plot(td_gen['width']['tracker_pos'][4], label='x tracker')
ax.set_title("Widened environment")

ax = fig.add_subplot(3, 2, 6)
ax.plot(td_targetblind_start['target_pos'][0], label='x target')
ax.plot(td_targetblind_start['tracker_pos'][0], label='x tracker')
ax.set_title("Target-blind agents")

ax.legend(loc="lower right", fontsize="small", markerscale=0.5, labelspacing=0.1)
plt.tight_layout()
plt.savefig('generalizations.eps')


"""
There are several key questions:
1. Does the tracker predict the target's reversal at the border or does it merely follow it step after step? 
- if the target disappears for some time (or vision is cut off), will the tracker continue moving? 
- does it differ depending on whether the target is moving towards the border, away from the border or right at the turn?
2. How is the turn at the border induced? Does the perception of the border affect the attractor?
3. How are the two agents jointly contributing to the movement pattern? 
4. Is there correlation between target movements and any of the agents nodes? - plot over time
5. Is there a predictive relationship or a correlation between the output of one agent and one of the nodes of the other?
- get a number?
"""


# catch just the agent brain networks
net1 = ag1.brain
net2 = ag2.brain

# reset the input to desired values, e.g. all 0
net1.I = np.zeros((8))
net2.I = np.zeros((8))

# reset initial state to all 0
net1.randomize_state([0, 0])
net2.randomize_state([0, 0])

sim_length = 5000

net_history = {'y1': np.zeros((sim_length, 8)), 'y2': np.zeros((sim_length, 8)),
               'o1': np.zeros((sim_length, 8)), 'o2': np.zeros((sim_length, 8)),
               'v_left': np.zeros((sim_length)), 'v_right': np.zeros((sim_length))}

for i in range(sim_length):
    net_history['y1'][i, :] = net1.Y
    net_history['y2'][i, :] = net2.Y
    net_history['o1'][i, :] = expit(net1.Y + net1.Theta)
    net_history['o2'][i, :] = expit(net2.Y + net2.Theta)
    a1o7 = net_history['o1'][i, 6]
    a1o8 = net_history['o1'][i, 7]
    a2o7 = net_history['o2'][i, 6]
    a2o8 = net_history['o2'][i, 7]
    activation_left = (a1o7 * ag1.MW[0] + a1o8 * ag1.MW[1]) * -1
    activation_right = a2o7 * ag2.MW[2] + a2o8 * ag2.MW[3]
    net_history['v_left'][i] = activation_left
    net_history['v_right'][i] = activation_right
    net1.euler_step()
    net2.euler_step()

plt.plot(net_history['v_left'], net_history['v_right'])
plt.plot(net_history['v_left'][0], net_history['v_right'][0], 'ro', markersize=7)
plt.plot(net_history['v_left'][-1], net_history['v_right'][-1], 'ro', markersize=7, mfc='none')


# Define the sample space (plotting ranges)
ymin = min(np.amin(net_history['y1'][:,6]), np.amin(net_history['y1'][:,7]))
ymax = max(np.amax(net_history['y1'][:,6]), np.amax(net_history['y1'][:,7]))

y_spread = np.linspace(ymin, ymax, 30)
Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8 = np.meshgrid(y_spread, y_spread, y_spread, y_spread,
                                             y_spread, y_spread, y_spread, y_spread)
dim_y = y1.shape[0]


"""
Define a helper function, which given a point in the state space, 
will tell us what the derivatives of the state elements will be.
One way to do this is to run the model over a single timestep, and extract the derivative information.
"""

net_input = np.zeros((8))

def get_derivative(network, state):
    # compute the next state of the network given its current state and the euler equation
    # update the outputs of all neurons
    o = expit(np.multiply(network.G, state + network.Theta))
    # update the state of all neurons
    dy_dt = np.multiply(1 / network.Tau, - state + np.dot(network.W, o) + net_input) * network.step_size
    state += dy_dt
    return tuple(dy_dt)

"""
Calculate the state space derivatives across our sample space.
"""

changes_y1 = np.zeros([dim_y, dim_y])
changes_y2 = np.zeros([dim_y, dim_y])

for i in range(dim_y):
    for j in range(dim_y):
        changes = get_derivative(net1, np.array([Y1[i,j], Y2[i,j]]))
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


td_steady = az.run_best_pair_steady(agent_directory, ag1, ag2)
td_steady2 = az.run_best_pair_steady(agent_directory, ag1, ag2)

positions = list(range(-20, 20))
# Define the sample space (plotting ranges)
ymin = min(positions)
ymax = max(positions)
num_points = len(positions)
# Define plotting grid
y1 = np.linspace(ymin, ymax, num_points)
y2 = np.linspace(ymin, ymax, num_points)
Y1, Y2 = np.meshgrid(y1, y2)

# overall vel
changes_y1 = np.zeros((num_points, num_points))
# fetch the velocities
for i in range(len(positions)):
    for j in range(len(positions)):
        changes_y1[i, j] = td_steady[positions[i]]['tracker_v'][j][-1]

changes_y2 = np.zeros((num_points, num_points))

# Plot the phase portrait
# quiver function takes the grid of x-y coordinates and their derivatives
plt.figure(figsize=(10, 6))
plt.quiver(Y1, Y2, changes_y1, changes_y2, color='b', alpha=.75)
plt.box('off')
plt.xlabel('target position', fontsize=14)
plt.ylabel('starting tracker position', fontsize=14)
# plt.show()
plt.savefig('velgrid.eps')


# left vel
changes_y1 = np.zeros((num_points, num_points))
# fetch the velocities
for i in range(len(positions)):
    for j in range(len(positions)):
        changes_y1[i, j] = td_steady[positions[i]]['keypress'][j][-1, 0]
changes_y2 = np.zeros((num_points, num_points))

# Plot the phase portrait
# quiver function takes the grid of x-y coordinates and their derivatives
plt.figure(figsize=(10, 6))
plt.quiver(Y1, Y2, changes_y1, changes_y2, color='g', alpha=.75)
plt.box('off')
plt.xlabel('target position', fontsize=14)
plt.ylabel('starting tracker position', fontsize=14)
plt.show()


# right vel
changes_y1 = np.zeros((num_points, num_points))
# fetch the velocities
for i in range(len(positions)):
    for j in range(len(positions)):
        changes_y1[i, j] = td_steady[positions[i]]['keypress'][j][-1, 1]

changes_y2 = np.zeros((num_points, num_points))

# Plot the phase portrait
# quiver function takes the grid of x-y coordinates and their derivatives
plt.figure(figsize=(10, 6))
plt.quiver(Y1, Y2, changes_y1, changes_y2, color='r', alpha=.75)
plt.box('off')
plt.xlabel('target position', fontsize=14)
plt.ylabel('starting tracker position', fontsize=14)
plt.show()



target_pos_dict = {}
for i in range(len(positions)):
    target_pos_dict[positions[i]] = i

# def plot_steady_vels(target_pos, tracker_pos):
#     plt.plot(td_steady[str(tracker_pos)]['tracker_v'][target_pos_dict[target_pos]],
#             label='Tg {}, Tr {}'.format(target_pos, tracker_pos))
# plot_steady_vels(0, -15)

fig = plt.figure(figsize=(10, 6))
targets = [0, -19, 19]
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    ax.plot(td_steady[str(-18)]['tracker_v'][target_pos_dict[targets[i]]],
                label='Tg {}, Tr {}'.format(targets[i], -18))
    ax.plot(td_steady[str(-15)]['tracker_v'][target_pos_dict[targets[i]]],
                label='Tg {}, Tr {}'.format(targets[i], -15))
    ax.plot(td_steady[str(0)]['tracker_v'][target_pos_dict[targets[i]]],
                label='Tg {}, Tr {}'.format(targets[i], 0))
    ax.plot(td_steady[str(15)]['tracker_v'][target_pos_dict[targets[i]]],
                label='Tg {}, Tr {}'.format(targets[i], 15))
    ax.plot(td_steady[str(18)]['tracker_v'][target_pos_dict[targets[i]]],
                label='Tg {}, Tr {}'.format(targets[i], 18))
    plt.legend()

print(td_steady["15"]['tracker_v'][39][-1])
print(td_steady["18"]['tracker_v'][39][-1])
print(td_steady["19"]['tracker_v'][39][-1])

plot_data(td, 3, "behavior", "Trial behavior")

fig = plt.figure(figsize=(12, 6))
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    ax.plot(td_steady[str(-18)]['keypress'][target_pos_dict[targets[i]]][:, 0], 'r',
                label='Tg {}, Tr {}, left'.format(targets[i], -18))
    ax.plot(td_steady[str(-18)]['keypress'][target_pos_dict[targets[i]]][:, 1], 'r--',
                label='Tg {}, Tr {}, right'.format(targets[i], -18))
    ax.plot(td_steady[str(-18)]['tracker_v'][target_pos_dict[targets[i]]], 'r:',
                label='Tg {}, Tr {}'.format(targets[i], -18))

    ax.plot(td_steady[str(0)]['keypress'][target_pos_dict[targets[i]]][:, 0],'b',
                label='Tg {}, Tr {}, left'.format(targets[i], 0))
    ax.plot(td_steady[str(0)]['keypress'][target_pos_dict[targets[i]]][:, 1],'b--',
                label='Tg {}, Tr {}, right'.format(targets[i], 0))
    ax.plot(td_steady[str(0)]['tracker_v'][target_pos_dict[targets[i]]], 'b:',
            label='Tg {}, Tr {}'.format(targets[i], 0))

    ax.plot(td_steady[str(18)]['keypress'][target_pos_dict[targets[i]]][:, 0],'g',
                label='Tg {}, Tr {}, left'.format(targets[i], 18))
    ax.plot(td_steady[str(18)]['keypress'][target_pos_dict[targets[i]]][:, 1],'g--',
                label='Tg {}, Tr {}, right'.format(targets[i], 18))
    ax.plot(td_steady[str(18)]['tracker_v'][target_pos_dict[targets[i]]],'g:',
                label='Tg {}, Tr {}'.format(targets[i], 18))

    ax.legend(bbox_to_anchor=(1.04,1), loc="lower center")

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 3, 1)
ax.plot(td_steady[str(-18)]['output_a1'][target_pos_dict[0]][:, 6], 'r',
            label='Tg {}, Tr {}, left_agent, n7'.format(0, -18))
ax.plot(td_steady[str(-18)]['output_a1'][target_pos_dict[0]][:, 7], 'r--',
            label='Tg {}, Tr {}, left_agent, n8'.format(0, -18))
ax.plot(td_steady[str(-18)]['output_a2'][target_pos_dict[0]][:, 6], 'b',
            label='Tg {}, Tr {}, right_agent, n7'.format(0, -18))
ax.plot(td_steady[str(-18)]['output_a2'][target_pos_dict[0]][:, 7], 'b--',
            label='Tg {}, Tr {}, right_agent, n8'.format(0, -18))
ax.legend(bbox_to_anchor=(1.04,1), loc="lower center")

ax = fig.add_subplot(1, 3, 2)
ax.plot(td_steady[str(-18)]['output_a1'][target_pos_dict[-19]][:, 6], 'r',
            label='Tg {}, Tr {}, left_agent, n7'.format(-19, -18))
ax.plot(td_steady[str(-18)]['output_a1'][target_pos_dict[-19]][:, 7], 'r--',
            label='Tg {}, Tr {}, left_agent, n8'.format(-19, -18))
ax.plot(td_steady[str(-18)]['output_a2'][target_pos_dict[-19]][:, 6], 'b',
            label='Tg {}, Tr {}, right_agent, n7'.format(-19, -18))
ax.plot(td_steady[str(-18)]['output_a2'][target_pos_dict[-19]][:, 7], 'b--',
            label='Tg {}, Tr {}, right_agent, n8'.format(-19, -18))
ax.legend(bbox_to_anchor=(1.04,1), loc="lower center")

ax = fig.add_subplot(1, 3, 3)
ax.plot(td_steady[str(18)]['output_a1'][target_pos_dict[19]][:, 6], 'r',
            label='Tg {}, Tr {}, left_agent, n7'.format(19, 18))
ax.plot(td_steady[str(18)]['output_a1'][target_pos_dict[19]][:, 7], 'r--',
            label='Tg {}, Tr {}, left_agent, n8'.format(19, 18))
ax.plot(td_steady[str(18)]['output_a2'][target_pos_dict[19]][:, 6], 'b',
            label='Tg {}, Tr {}, right_agent, n7'.format(19, 18))
ax.plot(td_steady[str(18)]['output_a2'][target_pos_dict[19]][:, 7], 'b--',
            label='Tg {}, Tr {}, right_agent, n8'.format(19, 18))
ax.legend(bbox_to_anchor=(1.04,1), loc="lower center")



td_immobile = az.run_best_pair_simple(agent_directory, ag1, ag2)
# plot_data(td_immobile, "all", "behavior", "Trial behavior")

num_trials = len(td_immobile[0]['target_pos'])
selected_positions = [-20, -15, -5, 0, 5, 15, 20]

fig = plt.figure(figsize=(14, 8))
for i in range(len(selected_positions)):
    ax = fig.add_subplot(2, 4, i+1)
    for trial_num in range(num_trials-1):
        plt.plot(td_immobile[selected_positions[i]]['target_pos'][trial_num] -
                 td_immobile[selected_positions[i]]['tracker_pos'][trial_num])

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(2, 2, 1)
ax.plot(td_immobile[0]['target_pos'][0] - td_immobile[0]['tracker_pos'][0])
ax.plot(td_immobile[0]['target_pos'][20] - td_immobile[0]['tracker_pos'][20])
ax.plot(td_immobile[0]['target_pos'][39] - td_immobile[0]['tracker_pos'][39])
ax = fig.add_subplot(2, 2, 2)
ax.plot(td_immobile[0]['target_pos'][0], label='Target position')
ax.plot(td_immobile[0]['tracker_pos'][0], label='Tracker position')
ax.plot(td_immobile[0]['tracker_v'][0], label='Tracker velocity')
ax.plot(td_immobile[0]['keypress'][0][:, 0], label='Left motor')
ax.plot(td_immobile[0]['keypress'][0][:, 1], label='Right motor')
ax = fig.add_subplot(2, 2, 3)
ax.plot(td_immobile[0]['target_pos'][20], label='Target position')
ax.plot(td_immobile[0]['tracker_pos'][20], label='Tracker position')
ax.plot(td_immobile[0]['tracker_v'][20], label='Tracker velocity')
ax.plot(td_immobile[0]['keypress'][20][:, 0], label='Left motor')
ax.plot(td_immobile[0]['keypress'][20][:, 1], label='Right motor')
ax = fig.add_subplot(2, 2, 4)
ax.plot(td_immobile[0]['target_pos'][39], label='Target position')
ax.plot(td_immobile[0]['tracker_pos'][39], label='Tracker position')
ax.plot(td_immobile[0]['tracker_v'][39], label='Tracker velocity')
ax.plot(td_immobile[0]['keypress'][39][:, 0], label='Left motor')
ax.plot(td_immobile[0]['keypress'][39][:, 1], label='Right motor')
ax.legend(bbox_to_anchor=(1.04,1), loc="lower center")


fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 3, 1)
ax.plot(td_immobile[0]['output_a1'][0][:, 6], 'r',
            label='Tg {}, Tr {}, left_agent, n7'.format(-20, 0))
ax.plot(td_immobile[0]['output_a1'][0][:, 7], 'r--',
            label='Tg {}, Tr {}, left_agent, n8'.format(-20, 0))
ax.plot(td_immobile[0]['output_a2'][0][:, 6], 'b',
            label='Tg {}, Tr {}, right_agent, n7'.format(-20, 0))
ax.plot(td_immobile[0]['output_a2'][0][:, 7], 'b--',
            label='Tg {}, Tr {}, right_agent, n8'.format(-20, 0))
ax.legend(bbox_to_anchor=(1.04,1), loc="lower center")

ax = fig.add_subplot(1, 3, 2)
ax.plot(td_immobile[0]['output_a1'][20][:, 6], 'r',
            label='Tg {}, Tr {}, left_agent, n7'.format(0, 0))
ax.plot(td_immobile[0]['output_a1'][20][:, 7], 'r--',
            label='Tg {}, Tr {}, left_agent, n8'.format(0, 0))
ax.plot(td_immobile[0]['output_a2'][20][:, 6], 'b',
            label='Tg {}, Tr {}, right_agent, n7'.format(0, 0))
ax.plot(td_immobile[0]['output_a2'][20][:, 7], 'b--',
            label='Tg {}, Tr {}, right_agent, n8'.format(0, 0))
ax.legend(bbox_to_anchor=(1.04,1), loc="lower center")

ax = fig.add_subplot(1, 3, 3)
ax.plot(td_immobile[0]['output_a1'][39][:, 6], 'r',
            label='Tg {}, Tr {}, left_agent, n7'.format(19, 0))
ax.plot(td_immobile[0]['output_a1'][39][:, 7], 'r--',
            label='Tg {}, Tr {}, left_agent, n8'.format(19, 0))
ax.plot(td_immobile[0]['output_a2'][39][:, 6], 'b',
            label='Tg {}, Tr {}, right_agent, n7'.format(19, 0))
ax.plot(td_immobile[0]['output_a2'][39][:, 7], 'b--',
            label='Tg {}, Tr {}, right_agent, n8'.format(19, 0))
ax.legend(bbox_to_anchor=(1.04,1), loc="lower center")




fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
hinton(ag1.MW, max_weight=10)
ax = fig.add_subplot(1, 2, 2)
hinton(ag2.MW, max_weight=10)


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
hinton(ag1.VW, max_weight=100)
ax = fig.add_subplot(1, 2, 2)
hinton(ag2.VW, max_weight=100)
print(ag2.VW)




def plot_visual_outputs_border(trial_data):
    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(14, 8))

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1)
        input_left_border = trial_data['input_a1'][p][:, 0] / ag1.VW[0] / 10
        input_right_border = trial_data['input_a1'][p][:, 1] / ag1.VW[1] / 10

        ax.plot(input_left_border, label='input left border')
        ax.plot(input_right_border, label='input right border')

        ax.plot(trial_data['output_a1'][p][:, 0], label='output ag1 n1')
        ax.plot(trial_data['output_a1'][p][:, 1], label='output ag1 n2')
        ax.plot(trial_data['output_a2'][p][:, 0], label='output ag2 n1', )
        ax.plot(trial_data['output_a2'][p][:, 1], label='output ag2 n2', )

    plt.legend(bbox_to_anchor=(1.04, 1), loc="lower center")
    plt.show()

plot_visual_outputs_border(td)


fig = plt.figure(figsize=(9, 6))

ax = fig.add_subplot(2, 2, 1)
ax.set_title("Border sensors input vs neuronal output")
input_left_border = td['input_a1'][5][:, 0] / ag1.VW[0] / 10
input_right_border = td['input_a1'][5][:, 1] / ag1.VW[1] / 10
ax.plot(input_left_border, label='left', alpha=0.5)
ax.plot(input_right_border, label='right', alpha=0.5)
ax.plot(td['output_a1'][5][:, 0], label='ag1 n1')
ax.plot(td['output_a1'][5][:, 1], label='ag1 n2')
ax.plot(td['output_a2'][5][:, 0], label='ag2 n1')
ax.plot(td['output_a2'][5][:, 1], label='ag2 n2')
ax.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1, framealpha=1)

ax = fig.add_subplot(2, 2, 2)
ax.set_title("Target sensors input vs neuronal output")
input_left_target = td['input_a1'][5][:, 2] / ag1.VW[2] / 10
input_right_target = td['input_a1'][5][:, 3] / ag1.VW[3] / 10
ax.plot(input_left_target, label='left', alpha=0.5)
ax.plot(input_right_target, label='right', alpha=0.5)
ax.plot(td['output_a1'][5][:, 2], label='ag1 n3')
ax.plot(td['output_a1'][5][:, 3], label='ag1 n4')
ax.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1, framealpha=1)

ax = fig.add_subplot(2, 2, 3)
ax.set_title("Target sensors input vs neuronal output")
input_left_target = td['input_a2'][5][:, 2] / ag2.VW[2] / 10
input_right_target = td['input_a2'][5][:, 3] / ag2.VW[3] / 10
ax.plot(input_left_target, label='left', alpha=0.5)
ax.plot(input_right_target, label='right', alpha=0.5)
ax.plot(td['output_a2'][5][:, 2], label='ag2 n3')
ax.plot(td['output_a2'][5][:, 3], label='ag2 n4')
ax.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1, framealpha=1)

ax = fig.add_subplot(2, 2, 4)
ax.set_title("Audio sensors input vs neuronal output")
input_left_audio = td['input_a1'][5][:, 4] / ag1.AW[0] / 10
input_right_audio = td['input_a1'][5][:, 5] / ag1.AW[1] / 10
ax.plot(input_left_audio, label='left', alpha=0.5)
ax.plot(input_right_audio, label='right', alpha=0.5)
ax.plot(td['output_a1'][5][:, 4], label='ag1 n5')
ax.plot(td['output_a1'][5][:, 5], label='ag1 n6')
ax.plot(td['output_a2'][5][:, 4], label='ag2 n5')
ax.plot(td['output_a2'][5][:, 5], label='ag2 n6')
ax.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1, framealpha=1)

plt.tight_layout()
plt.savefig('inputs.eps')



def plot_visual_outputs_target(trial_data):
    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(14, 8))

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1)
        ax.plot(trial_data['output_a1'][p][:, 2], label='n3')
        ax.plot(trial_data['output_a1'][p][:, 3], label='n4')
        ax.plot(trial_data['output_a2'][p][:, 2], label='n3', ls='--')
        ax.plot(trial_data['output_a2'][p][:, 3], label='n4', ls='--')

    plt.legend(bbox_to_anchor=(1.04, 1), loc="lower center")
    plt.show()


def plot_audio_outputs(trial_data):
    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(14, 8))

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1)
        ax.plot(trial_data['output_a1'][p][:, 4], label='n5')
        ax.plot(trial_data['output_a1'][p][:, 5], label='n6')
        ax.plot(trial_data['output_a2'][p][:, 4], label='n5', ls='--')
        ax.plot(trial_data['output_a2'][p][:, 5], label='n6', ls='--')

    plt.legend(bbox_to_anchor=(1.04, 1), loc="lower center")
    plt.show()


def plot_motor_outputs(trial_data):
    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(14, 8))

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1)
        ax.plot(trial_data['output_a1'][p][:, 6], label='n7')
        ax.plot(trial_data['output_a1'][p][:, 7], label='n8')
        ax.plot(trial_data['output_a2'][p][:, 6], label='n7', ls='--')
        ax.plot(trial_data['output_a2'][p][:, 7], label='n8', ls='--')

    plt.legend(bbox_to_anchor=(1.04, 1), loc="lower center")
    plt.show()


plot_visual_outputs_target(td)
plot_audio_outputs(td)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
hinton(ag1.AW, max_weight=100)
ax = fig.add_subplot(1, 2, 2)
hinton(ag2.AW, max_weight=100)
plot_motor_outputs(td)


def plot_activation_all(trial_data, agent_num):
    k = 'brain_state_a' + str(agent_num)
    num_trials = len(trial_data['target_pos'])
    num_cols = num_trials / 2
    fig = plt.figure(figsize=(14, 8))

    for p in range(num_trials):
        ax = fig.add_subplot(2, num_cols, p + 1)
        ax.plot(trial_data[k][p][:, 0], label='n1')
        ax.plot(trial_data[k][p][:, 1], label='n2')
        ax.plot(trial_data[k][p][:, 2], label='n3')
        ax.plot(trial_data[k][p][:, 3], label='n4')
        ax.plot(trial_data[k][p][:, 4], label='n5')
        ax.plot(trial_data[k][p][:, 5], label='n6')
        ax.plot(trial_data[k][p][:, 6], label='n7')
        ax.plot(trial_data[k][p][:, 7], label='n8')

    plt.legend(bbox_to_anchor=(1.04, 1), loc="lower center")
    plt.show()


plot_activation_all(td, 1)

plot_activation_all(td, 2)




""" Information dynamics plots """


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]


def plot_inf(local, source, neuron_group, real_trial=False):
    num_destinations = len(local[source])
    num_cols = num_destinations/2
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Local measures for {}".format(source))
    for p in range(num_destinations):
        destination = list(local[source].keys())[p]
        data = local[source][destination]
        ax = fig.add_subplot(2, num_cols, p + 1)
        ax.set_title(destination)
        if neuron_group == 'visual':
            ax.plot(list(smooth(np.array(data[0]), 25, 'blackman')), label='n1')
            ax.plot(list(smooth(np.array(data[1]), 25, 'blackman')), label='n2')
            ax.plot(list(smooth(np.array(data[2]), 25, 'blackman')), label='n3')
            ax.plot(list(smooth(np.array(data[3]), 25, 'blackman')), label='n4')
        elif neuron_group == 'auditory':
            ax.plot(list(smooth(np.array(data[4]), 25, 'blackman')), label='n5')
            ax.plot(list(smooth(np.array(data[5]), 25, 'blackman')), label='n6')
        elif neuron_group == 'motor':
            ax.plot(list(smooth(np.array(data[6]), 25, 'blackman')), label='n7')
            ax.plot(list(smooth(np.array(data[7]), 25, 'blackman')), label='n8')
        else:
            ax.plot(list(smooth(np.array(data[0][:3050]), 25, 'blackman')), label='n1')
            ax.plot(list(smooth(np.array(data[1][:3050]), 25, 'blackman')), label='n2')
            ax.plot(list(smooth(np.array(data[2][:3050]), 25, 'blackman')), label='n3')
            ax.plot(list(smooth(np.array(data[3][:3050]), 25, 'blackman')), label='n4')
            ax.plot(list(smooth(np.array(data[4][:3050]), 25, 'blackman')), label='n5')
            ax.plot(list(smooth(np.array(data[5][:3050]), 25, 'blackman')), label='n6')
            ax.plot(list(smooth(np.array(data[6][:3050]), 25, 'blackman')), label='n7')
            ax.plot(list(smooth(np.array(data[7][:3050]), 25, 'blackman')), label='n8')

        if real_trial:
            ax.axvline(x=1600, color='black')
            ax.axvline(x=3100, color='black')
            ax.axvline(x=4600, color='black')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()


""" TE """

pkl_file = open('te_trial1.pkl', 'rb')
te1 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('te_trial2.pkl', 'rb')
te2 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('te_trial3.pkl', 'rb')
te3 = pickle.load(pkl_file)
pkl_file.close()


pkl_file = open('te_trial4.pkl', 'rb')
te4 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('te_trial5.pkl', 'rb')
te5 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('te_trial6.pkl', 'rb')
te6 = pickle.load(pkl_file)
pkl_file.close()


def create_avg_te(tes):
    te_global_avg = deepcopy(tes[0][0])
    te_local_avg = deepcopy(tes[0][1])

    for source in te_local_avg:
        for destination in te_local_avg[source]:
            te_global_avg[source][destination] = np.mean([tes[0][0][source][destination],
                                                          tes[1][0][source][destination],
                                                          tes[2][0][source][destination],
                                                          tes[3][0][source][destination],
                                                          tes[4][0][source][destination],
                                                          tes[5][0][source][destination]],
                                                         axis=0)
            for i in range(8):
                te_local_avg[source][destination][i] = np.mean([tes[0][1][source][destination][i],
                                                                tes[1][1][source][destination][i],
                                                                tes[2][1][source][destination][i],
                                                                tes[3][1][source][destination][i],
                                                                tes[4][1][source][destination][i],
                                                                tes[5][1][source][destination][i]],
                                                               axis=0)
    return te_global_avg, te_local_avg


avg_global_tes, avg_local_tes = create_avg_te([te1, te2, te3, te4, te5, te6])

# params_te = te[2]
# print(params_te)

for te_source in avg_global_tes:
    for te_destination in avg_global_tes[te_source]:
        print("TE for {} to {}:".format(te_source, te_destination))
        to_print = [num if (num < -0.01 or num > 0.01) else 0 for num in avg_global_tes[te_source][te_destination] ]
        print(to_print)
        print("\n")

plot_inf(avg_local_tes, 'target_dist', 'all')
plot_inf(avg_local_tes, 'target_dist', 'visual')
plot_inf(avg_local_tes, 'target_dist', 'auditory')
plot_inf(avg_local_tes, 'target_dist', 'motor')


# particular example
to_plot = [num if (num < -0.25 or num > 1) else 0 for num in te3[1]['target_dist']['outputs_a1'][3]]
plt.plot(resampled_td['target_pos'][2]/20)
plt.plot(resampled_td['tracker_pos'][2]/20)
for i in range(len(to_plot)):
    te_point = to_plot[i]
    if te_point < 0:
        plt.axvline(x=i, color='blue')
    elif te_point > 0:
        plt.axvline(x=i, color='red')


""" MI """

pkl_file = open('mi_trial1.pkl', 'rb')
mi1 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('mi_trial2.pkl', 'rb')
mi2 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('mi_trial3.pkl', 'rb')
mi3 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('mi_trial4.pkl', 'rb')
mi4 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('mi_trial5.pkl', 'rb')
mi5 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('mi_trial6.pkl', 'rb')
mi6 = pickle.load(pkl_file)
pkl_file.close()

avg_global_mis, avg_local_mis = create_avg_te([mi1, mi2, mi3, mi4, mi5, mi6])

for mi_source in avg_global_mis:
    for mi_destination in avg_global_mis[mi_source]:
        print("MI for {} to {}:".format(mi_source, mi_destination))
        to_print = [num if (num < -0.01 or num > 0.01) else 0 for num in avg_global_mis[mi_source][mi_destination] ]
        print(to_print)
        print("\n")

plot_inf(avg_local_mis, 'target_dist', 'all')
plot_inf(avg_local_mis, 'target_dist', 'visual')
plot_inf(avg_local_mis, 'target_dist', 'auditory')
plot_inf(avg_local_mis, 'target_dist', 'motor')

plot_inf(mi3[1], 'target_dist', 'visual')

""" AIS """


def plot_ais(local, neuron_group, real_trial=False):
    num_destinations = len(list(local.keys()))
    num_cols = num_destinations/2
    fig = plt.figure(figsize=(10, 6))

    for p in range(num_destinations):
        destination = list(local.keys())[p]
        data = local[destination]
        ax = fig.add_subplot(2, num_cols, p + 1)
        ax.set_title(destination)
        if neuron_group == 'visual':
            ax.plot(list(smooth(np.array(data[0]), 25, 'blackman')), label='n1')
            ax.plot(list(smooth(np.array(data[1]), 25, 'blackman')), label='n2')
            ax.plot(list(smooth(np.array(data[2]), 25, 'blackman')), label='n3')
            ax.plot(list(smooth(np.array(data[3]), 25, 'blackman')), label='n4')
        elif neuron_group == 'auditory':
            ax.plot(list(smooth(np.array(data[4]), 25, 'blackman')), label='n5')
            ax.plot(list(smooth(np.array(data[5]), 25, 'blackman')), label='n6')
        elif neuron_group == 'motor':
            ax.plot(list(smooth(np.array(data[6]), 25, 'blackman')), label='n7')
            ax.plot(list(smooth(np.array(data[7]), 25, 'blackman')), label='n8')
        else:
            for i in range(8):
                ax.plot(list(smooth(np.array(data[i]), 25, 'blackman')), label='n{}'.format(i+1))

        if real_trial:
            ax.axvline(x=1600, color='black')
            ax.axvline(x=3100, color='black')
            ax.axvline(x=4600, color='black')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()


def create_avg_ais(aises):
    te_global_avg = deepcopy(aises[0][0])
    te_local_avg = deepcopy(aises[0][1])

    for destination in te_local_avg:
        te_global_avg[destination] = np.mean([aises[0][0][destination],
                                                      aises[1][0][destination],
                                                      aises[2][0][destination],
                                                      aises[3][0][destination],
                                                      aises[4][0][destination],
                                                      aises[5][0][destination]],
                                                     axis=0)
        for i in range(8):
            te_local_avg[destination][i] = np.mean([aises[0][1][destination][i],
                                                            aises[1][1][destination][i],
                                                            aises[2][1][destination][i],
                                                            aises[3][1][destination][i],
                                                            aises[4][1][destination][i],
                                                            aises[5][1][destination][i]],
                                                           axis=0)
    return te_global_avg, te_local_avg


pkl_file = open('ais_trial1.pkl', 'rb')
ais1 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('ais_trial2.pkl', 'rb')
ais2 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('ais_trial3.pkl', 'rb')
ais3 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('ais_trial4.pkl', 'rb')
ais4 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('ais_trial5.pkl', 'rb')
ais5 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('ais_trial6.pkl', 'rb')
ais6 = pickle.load(pkl_file)
pkl_file.close()

avg_global_ais, avg_local_ais = create_avg_ais([ais1, ais2, ais3, ais4, ais5, ais6])

for ais_destination in avg_global_ais:
    print("AIS for {}:".format(ais_destination))
    to_print = [num if (num < -0.01 or num > 0.01) else 0 for num in avg_global_ais[ais_destination] ]
    print(to_print)
    print("\n")

# plot_ais(avg_local_ais, 'all')
plot_ais(avg_local_ais, 'visual')
plot_ais(avg_local_ais, 'auditory')
plot_ais(avg_local_ais, 'motor')

plot_ais(ais3[1], 'visual')
plot_ais(ais1[1], 'visual')
plot_ais(ais6[1], 'visual')

plt.plot(list(smooth(np.array(ais3[1]['outputs_a1'][3]), 25, 'blackman')))
plt.plot(list(smooth(np.array(te3[1]['target_dist']['outputs_a1'][3]), 25, 'blackman')))


plt.plot(list(smooth(np.array(ais3[1]['outputs_a1'][6]), 25, 'blackman')))
plt.plot(list(smooth(np.array(te3[1]['target_dist']['outputs_a1'][6]), 25, 'blackman')))

""" Cross MI """


def plot_cross_mi(local, neuron_group, real_trial=False):
    num_sources = len(list(local.keys()))
    num_cols = num_sources
    fig = plt.figure(figsize=(12, 6))

    for p in range(num_sources):
        source = list(local.keys())[p]
        data = local[source]
        ax = fig.add_subplot(1, num_cols, p + 1)
        ax.set_title(source)
        if neuron_group == 'visual':
            for i in range(4):
                ax.plot(list(smooth(np.array(data[i]), 25, 'blackman')), label='n{}'.format(i+1))
        elif neuron_group == 'auditory':
            for i in range(4, 6):
                ax.plot(list(smooth(np.array(data[i]), 25, 'blackman')), label='n{}'.format(i+1))
        elif neuron_group == 'motor':
            for i in range(6, 8):
                ax.plot(list(smooth(np.array(data[i]), 25, 'blackman')), label='n{}'.format(i+1))
        else:
            for i in range(8):
                ax.plot(list(smooth(np.array(data[i]), 25, 'blackman')), label='n{}'.format(i+1))

        if real_trial:
            ax.axvline(x=1600, color='black')
            ax.axvline(x=3100, color='black')
            ax.axvline(x=4600, color='black')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


def create_avg_cross_mi(crossmisses):
    te_global_avg = deepcopy(crossmisses[0][0])
    te_local_avg = deepcopy(crossmisses[0][1])

    for source in te_local_avg:
        te_global_avg[source] = np.mean([crossmisses[0][0][source],
                                              crossmisses[1][0][source],
                                              crossmisses[2][0][source],
                                              crossmisses[3][0][source],
                                              crossmisses[4][0][source],
                                              crossmisses[5][0][source]],
                                                     axis=0)
        for i in range(8):
            te_local_avg[source][i] = np.mean([crossmisses[0][1][source][i],
                                                    crossmisses[1][1][source][i],
                                                    crossmisses[2][1][source][i],
                                                    crossmisses[3][1][source][i],
                                                    crossmisses[4][1][source][i],
                                                    crossmisses[5][1][source][i]],
                                                           axis=0)
    return te_global_avg, te_local_avg


pkl_file = open('cross_mi_trial1.pkl', 'rb')
cross_mi1 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('cross_mi_trial2.pkl', 'rb')
cross_mi2 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('cross_mi_trial3.pkl', 'rb')
cross_mi3 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('cross_mi_trial4.pkl', 'rb')
cross_mi4 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('cross_mi_trial5.pkl', 'rb')
cross_mi5 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('cross_mi_trial6.pkl', 'rb')
cross_mi6 = pickle.load(pkl_file)
pkl_file.close()

avg_global_ais21, avg_local_ais21 = create_avg_cross_mi([cross_mi1[0], cross_mi2[0], cross_mi3[0],
                                                     cross_mi4[0], cross_mi5[0], cross_mi6[0]])

avg_global_ais12, avg_local_ais12 = create_avg_cross_mi([cross_mi1[1], cross_mi2[1], cross_mi3[1],
                                                     cross_mi4[1], cross_mi5[1], cross_mi6[1]])

for source2 in avg_global_ais21:
    print("MI for {} in agent 2 to action of agent 1:".format(source2))
    to_print = [num if (num < -0.01 or num > 0.01) else 0 for num in avg_global_ais21[source2]]
    print(to_print)
    print("\n")

for source1 in avg_global_ais12:
    print("MI for {} in agent 1 to action of agent 2:".format(source1))
    to_print = [num if (num < -0.01 or num > 0.01) else 0 for num in avg_global_ais12[source1]]
    print(to_print)
    print("\n")

plot_cross_mi(avg_local_ais21, 'all')
plot_cross_mi(avg_local_ais12, 'all')

plot_cross_mi(cross_mi3[0][1], 'auditory')
plot_cross_mi(cross_mi3[0][1], 'motor')


# TODO steady state?
# TODO different cross mi lags?
# TODO significance testing

# td_immobile = az.run_best_pair_simple(agent_directory, ag1, ag2)
# output = open('td_immobile_914463.pkl', 'wb')
# pickle.dump(td_immobile, output)
# output.close()


""" Immobile target data"""

fig = plt.figure(figsize=(12, 6))
plt.plot(td_immobile[0]['target_pos'][30])
plt.plot(td_immobile[0]['tracker_pos'][30])


""" TE """

pkl_file = open('te_tg15_tr0_914463.pkl', 'rb')
te = pickle.load(pkl_file)
pkl_file.close()

local_te = te[1]

plot_inf(local_te, 'target_dist', 'all')
plot_inf(local_te, 'target_dist', 'visual')
plot_inf(local_te, 'target_dist', 'auditory')
plot_inf(local_te, 'target_dist', 'motor')


plot_inf(local_te, 'left_border_dist', 'all')
plot_inf(local_te, 'right_border_dist', 'all')


""" MI """

pkl_file = open('mi_tg15_tr0_914463.pkl', 'rb')
mi = pickle.load(pkl_file)
pkl_file.close()

local_mi = mi[1]

plot_inf(local_mi, 'target_dist', 'all')
plot_inf(local_mi, 'target_dist', 'visual')
plot_inf(local_mi, 'target_dist', 'auditory')
plot_inf(local_mi, 'target_dist', 'motor')

plot_inf(local_mi, 'left_border_dist', 'all')
plot_inf(local_mi, 'right_border_dist', 'all')


""" AIS """

pkl_file = open('ais_tg15_tr0_914463.pkl', 'rb')
ais = pickle.load(pkl_file)
pkl_file.close()

local_ais = ais[1]

plot_ais(local_ais, 'all')
plot_ais(local_ais, 'visual')
plot_ais(local_ais, 'auditory')
plot_ais(local_ais, 'motor')


""" Cross MI """

pkl_file = open('cross_mi_tg15_tr0_914463.pkl', 'rb')
cross_mi = pickle.load(pkl_file)
pkl_file.close()

brain2_to_action1_avg = cross_mi[0][0]
brain2_to_action1_local = cross_mi[0][1]
brain1_to_action2_avg = cross_mi[1][0]
brain1_to_action2_local = cross_mi[1][1]

plot_cross_mi(brain2_to_action1_local, 'all')
plot_cross_mi(brain2_to_action1_local, 'visual')
plot_cross_mi(brain2_to_action1_local, 'auditory')
plot_cross_mi(brain2_to_action1_local, 'motor')

plot_cross_mi(brain1_to_action2_local, 'all')
plot_cross_mi(brain1_to_action2_local, 'visual')
plot_cross_mi(brain1_to_action2_local, 'auditory')
plot_cross_mi(brain1_to_action2_local, 'motor')



def dynamics_all_neurons(trial_num):
    target_dist = (td['target_pos'][trial_num] - td['tracker_pos'][trial_num]) / 10
    border_dist = (20 - td['tracker_pos'][trial_num]) / 20
    left_vel = (td['keypress'][trial_num][:, 0]) / 10
    right_vel = (td['keypress'][trial_num][:, 1]) / 10

    for i in range(8):
        plt.figure()
        plt.plot(list(smooth(np.array(all_aises[trial_num][1]['activation_a1'][i]), 25, 'blackman')), label='ais')
        # plt.plot(list(smooth(np.array(all_tes[trial_num][1]['target_dist']['activation_a1'][i]), 25, 'blackman')), label='te target')
        # plt.plot(list(smooth(np.array(all_tes[trial_num][1]['left_border_dist']['activation_a1'][i]), 25, 'blackman')), label='te border')
        # plt.plot(list(smooth(np.array(all_tes[trial_num][1]['left_motor']['activation_a1'][i]), 25, 'blackman')),
        #          label='left ear')
        # plt.plot(list(smooth(np.array(all_tes[trial_num][1]['right_motor']['activation_a1'][i]), 25, 'blackman')),
        #          label='right ear')

        # plt.plot(list(smooth(np.array(all_aises[trial_num][1]['outputs_a1'][i]), 25, 'blackman')), label='ais')
        # plt.plot(list(smooth(np.array(all_tes[trial_num][1]['target_dist']['outputs_a1'][i]), 25, 'blackman')), label='te target')
        # plt.plot(list(smooth(np.array(all_tes[trial_num][1]['left_border_dist']['outputs_a1'][i]), 25, 'blackman')), label='te border')
        # plt.plot(list(smooth(np.array(all_tes[trial_num][1]['left_motor']['outputs_a1'][i]), 25, 'blackman')),
        #          label='left ear')
        # plt.plot(list(smooth(np.array(all_tes[trial_num][1]['right_motor']['outputs_a1'][i]), 25, 'blackman')),
        #          label='right ear')

        plt.plot(target_dist, label='target')
        plt.plot(border_dist, label='border')
        # plt.plot(left_vel, label='left vel')
        # plt.plot(right_vel, label='right vel')
        plt.legend()


def dynamics_all_trials(neuron_num):
    for trial_num in range(6):
        plt.figure()
        plt.plot(list(smooth(np.array(all_aises[trial_num][1]['outputs_a1'][neuron_num]), 25, 'blackman')), label='ais')
        plt.plot(list(smooth(np.array(all_tes[trial_num][1]['target_dist']['outputs_a1'][neuron_num]), 25, 'blackman')), label='te target')
        plt.plot(list(smooth(np.array(all_tes[trial_num][1]['left_border_dist']['outputs_a1'][neuron_num]), 25, 'blackman')),
                 label='te border')
        plt.legend()


dynamics_all_neurons(2)
dynamics_all_trials(1)


def plot_vel_infos(trial_num):
    target_dist = td_rs['target_pos'][trial_num] - td_rs['tracker_pos'][trial_num]
    vel_left = td_rs['keypress'][trial_num][:, 0]
    plt.figure()
    plt.plot(list(smooth(np.array(all_aises[trial_num][1]['activation_a1'][4]), 25, 'blackman')), label='n5')
    plt.plot(list(smooth(np.array(all_aises[trial_num][1]['activation_a1'][6]), 25, 'blackman')), label='n7')
    plt.plot(list(smooth(np.array(all_aises[trial_num][1]['activation_a1'][7]), 25, 'blackman')), label='n8')
    plt.plot(target_dist, label='target dist')
    plt.plot(vel_left, label='left velocity')
    plt.legend()

plot_vel_infos(2)



def create_avg_cross_mi(crossmisses):
    te_global_avg = deepcopy(crossmisses[0][0])
    te_local_avg = deepcopy(crossmisses[0][1])

    for source in te_local_avg:
        te_global_avg[source] = np.mean([crossmisses[0][0][source],
                                         crossmisses[1][0][source],
                                         crossmisses[2][0][source],
                                         crossmisses[3][0][source],
                                         crossmisses[4][0][source],
                                         crossmisses[5][0][source]], axis=0)
        for i in range(8):
            te_local_avg[source][i] = np.mean([crossmisses[0][1][source][i],
                                               crossmisses[1][1][source][i],
                                               crossmisses[2][1][source][i],
                                               crossmisses[3][1][source][i],
                                               crossmisses[4][1][source][i],
                                               crossmisses[5][1][source][i]], axis=0)
    return te_global_avg, te_local_avg


def load_crossmis():
    pkl_file = open('cross_mi_trial1.pkl', 'rb')
    cross_mi1 = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('cross_mi_trial2.pkl', 'rb')
    cross_mi2 = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('cross_mi_trial3.pkl', 'rb')
    cross_mi3 = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('cross_mi_trial4.pkl', 'rb')
    cross_mi4 = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('cross_mi_trial5.pkl', 'rb')
    cross_mi5 = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('cross_mi_trial6.pkl', 'rb')
    cross_mi6 = pickle.load(pkl_file)
    pkl_file.close()

    cross_mis = {'cross21': [cross_mi1[0], cross_mi2[0], cross_mi3[0], cross_mi4[0], cross_mi5[0], cross_mi6[0]],
                 'cross12': [cross_mi1[1], cross_mi2[1], cross_mi3[1], cross_mi4[1], cross_mi5[1], cross_mi6[1]]}
    return cross_mis


crossmis = load_crossmis()
avg_global_crmi21, avg_local_crmi21 = create_avg_cross_mi(crossmis['cross21'])
avg_global_crmi12, avg_local_crmi12 = create_avg_cross_mi(crossmis['cross12'])
