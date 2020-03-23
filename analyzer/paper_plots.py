import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import signal
from copy import deepcopy
import seaborn as sns
# import idtxl.visualise_graph as vg
# import networkx as nx
from ..analyzer import analyze as az
from ..python_simulator import simulate

data_directory = "processed_data/"

figure_directory = "paperfigs/"
sns.set_style('ticks')

agent_directory = "agents/joint/direct/914463"
gen_files = fnmatch.filter(os.listdir(agent_directory), 'gen*')
gen_numbers = [int(x[3:]) for x in gen_files]
last_gen = max(gen_numbers)
config = az.load_config(agent_directory)

# fitness
fit_file = open(agent_directory + '/fitnesses', 'rb')
fits = pickle.load(fit_file)
fit_file.close()

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)

ax.plot(fits['average'], label="Average population fitness")
ax.plot(fits['best'], label="Best agent fitness")
ax.set_title('Fitness over generations', fontsize=16)
ax.legend(loc="upper left", fontsize="medium", markerscale=0.5, labelspacing=0.1)
plt.tight_layout()
sns.despine()
plt.savefig(figure_directory + 'fitness.eps')

td, ag1, ag2 = az.run_best_pair(agent_directory, last_gen)

# # save data
# output = open('td_914463.pkl', 'wb')
# pickle.dump(td, output)
# output.close()

agent_directory_176 = "Agents/joint/direct/random/176176"
gen_files_176 = fnmatch.filter(os.listdir(agent_directory_176), 'gen*')
gen_numbers_176 = [int(x[3:]) for x in gen_files_176]
last_gen_176 = max(gen_numbers_176)
config_176 = az.load_config(agent_directory_176)
td_176, ag1_176, ag2_176 = az.run_best_pair(agent_directory_176, last_gen_176)

# output = open('td_176176.pkl', 'wb')
# pickle.dump(td_176, output)
# output.close()


# 2 strategies comparison
fig = plt.figure(figsize=(10, 4))

ax = fig.add_subplot(1, 2, 1)
ax.plot(td['target_pos'][1], label='x target')
ax.plot(td['tracker_pos'][1], label='x tracker')
ax.plot(td['tracker_v'][1], label='v tracker')
ax.plot(td['keypress'][1][:, 0], label='v left')
ax.plot(td['keypress'][1][:, 1], label='v right')
ax.set_title("A. Joint strategy", fontsize=16)
ax.legend(loc="upper right", fontsize="medium", markerscale=0.5, labelspacing=0.1)

ax = fig.add_subplot(1, 2, 2)
ax.plot(td_176['target_pos'][1], label='x target')
ax.plot(td_176['tracker_pos'][1], label='x tracker')
ax.plot(td_176['tracker_v'][1], label='v tracker')
ax.plot(td_176['keypress'][1][:, 0], label='v left')
ax.plot(td_176['keypress'][1][:, 1], label='v right')
ax.set_title("B. Independent strategy", fontsize=16)

sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'strategies.eps')


def resample_trials(trial_data):
    num_trials = len(trial_data['target_pos'])
    sampled_td = deepcopy(trial_data)

    for trial_num in range(num_trials):
        sampled_td['target_pos'][trial_num] = np.concatenate(
            (trial_data['target_pos'][trial_num][:100],
             signal.resample(trial_data['target_pos'][trial_num][100:], 3000)))
        sampled_td['tracker_pos'][trial_num] = np.concatenate(
            (trial_data['tracker_pos'][trial_num][:100],
             signal.resample(trial_data['tracker_pos'][trial_num][100:], 3000)))
        sampled_td['tracker_v'][trial_num] = np.concatenate(
            (trial_data['tracker_v'][trial_num][:100],
             signal.resample(trial_data['tracker_v'][trial_num][100:], 3000)))

        sampled_td['brain_state_a1'][trial_num] = np.zeros((3100, 8))
        sampled_td['derivatives_a1'][trial_num] = np.zeros((3100, 8))
        sampled_td['input_a1'][trial_num] = np.zeros((3100, 8))
        sampled_td['output_a1'][trial_num] = np.zeros((3100, 8))

        sampled_td['brain_state_a2'][trial_num] = np.zeros((3100, 8))
        sampled_td['derivatives_a2'][trial_num] = np.zeros((3100, 8))
        sampled_td['input_a2'][trial_num] = np.zeros((3100, 8))
        sampled_td['output_a2'][trial_num] = np.zeros((3100, 8))

        sampled_td['keypress'][trial_num] = np.zeros((3100, 2))
        sampled_td['button_state_a1'][trial_num] = np.zeros((3100, 2))
        sampled_td['button_state_a2'][trial_num] = np.zeros((3100, 2))

        for i in range(8):
            sampled_td['brain_state_a1'][trial_num][:, i] = np.concatenate(
                (trial_data['brain_state_a1'][trial_num][:100, i],
                 signal.resample(trial_data['brain_state_a1'][trial_num][100:, i], 3000)))
            sampled_td['derivatives_a1'][trial_num][:, i] = np.concatenate(
                (trial_data['derivatives_a1'][trial_num][:100, i],
                 signal.resample(trial_data['derivatives_a1'][trial_num][100:, i], 3000)))
            sampled_td['input_a1'][trial_num][:, i] = np.concatenate(
                (trial_data['input_a1'][trial_num][:100, i],
                 signal.resample(trial_data['input_a1'][trial_num][100:, i], 3000)))
            sampled_td['output_a1'][trial_num][:, i] = np.concatenate(
                (trial_data['output_a1'][trial_num][:100, i],
                 signal.resample(trial_data['output_a1'][trial_num][100:, i], 3000)))

            sampled_td['brain_state_a2'][trial_num][:, i] = np.concatenate(
                (trial_data['brain_state_a2'][trial_num][:100, i],
                 signal.resample(trial_data['brain_state_a2'][trial_num][100:, i], 3000)))
            sampled_td['derivatives_a2'][trial_num][:, i] = np.concatenate(
                (trial_data['derivatives_a2'][trial_num][:100, i],
                 signal.resample(trial_data['derivatives_a2'][trial_num][100:, i], 3000)))
            sampled_td['input_a2'][trial_num][:, i] = np.concatenate(
                (trial_data['input_a2'][trial_num][:100, i],
                 signal.resample(trial_data['input_a2'][trial_num][100:, i], 3000)))
            sampled_td['output_a2'][trial_num][:, i] = np.concatenate(
                (trial_data['output_a2'][trial_num][:100, i],
                 signal.resample(trial_data['output_a2'][trial_num][100:, i], 3000)))

        for i in range(2):
            sampled_td['keypress'][trial_num][:, i] = np.concatenate(
                (trial_data['keypress'][trial_num][:100, i],
                 signal.resample(trial_data['keypress'][trial_num][100:, i], 3000)))
            sampled_td['button_state_a1'][trial_num][:, i] = np.concatenate(
                (trial_data['button_state_a1'][trial_num][:100, i],
                 signal.resample(trial_data['button_state_a1'][trial_num][100:, i], 3000)))
            sampled_td['button_state_a2'][trial_num][:, i] = np.concatenate(
                (trial_data['button_state_a2'][trial_num][:100, i],
                 signal.resample(trial_data['button_state_a2'][trial_num][100:, i], 3000)))

    return sampled_td


# resample to the same trial length
resampled_td = resample_trials(td)

# output = open('resampled_td_914463.pkl', 'wb')
# pickle.dump(resampled_td, output)
# output.close()

# plot just trial behavior
num_trials = 6
lims = (-20, 20)

fig = plt.figure(figsize=(15, 8))

for p in range(num_trials):
    ax = fig.add_subplot(2, 3, p+1)
    ax.set_ylim(lims)
    ax.plot(resampled_td['target_pos'][p], label='x target')
    ax.plot(resampled_td['tracker_pos'][p], label='x tracker')
    ax.plot(resampled_td['tracker_v'][p], label='v tracker')
    ax.plot(resampled_td['keypress'][p][:, 0], label='v left')
    ax.plot(resampled_td['keypress'][p][:, 1], label='v right')

ax.legend(loc="lower right", fontsize="small", markerscale=0.5, labelspacing=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'trial_behavior.eps')


# make sure the agents can't accomplish the task on their own
td_left = az.run_agent_from_best_pair(agent_directory, last_gen, 'left')
td_right = az.run_agent_from_best_pair(agent_directory, last_gen, 'right')


def resample_behavior(trial_data):
    num_trials = len(trial_data['target_pos'])
    sampled_td = deepcopy(trial_data)
    for trial_num in range(num_trials):
        sampled_td['target_pos'][trial_num] = np.concatenate(
            (trial_data['target_pos'][trial_num][:100],
             signal.resample(trial_data['target_pos'][trial_num][100:], 3000)))
        sampled_td['tracker_pos'][trial_num] = np.concatenate(
            (trial_data['tracker_pos'][trial_num][:100],
             signal.resample(trial_data['tracker_pos'][trial_num][100:], 3000)))
        sampled_td['tracker_v'][trial_num] = np.concatenate(
            (trial_data['tracker_v'][trial_num][:100],
             signal.resample(trial_data['tracker_v'][trial_num][100:], 3000)))
        sampled_td['keypress'][trial_num] = np.zeros((3100, 2))
        for i in range(2):
            sampled_td['keypress'][trial_num][:, i] = np.concatenate(
                (trial_data['keypress'][trial_num][:100, i],
                 signal.resample(trial_data['keypress'][trial_num][100:, i], 3000)))
    return sampled_td


resampled_left = resample_behavior(td_left)
resampled_right = resample_behavior(td_right)

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 2, 1)
ax.set_ylim(lims)
ax.plot(resampled_left['target_pos'][0], label='x target')
ax.plot(resampled_left['tracker_pos'][0], label='x tracker tr 1')
ax.plot(resampled_left['tracker_pos'][1], label='x tracker tr 2')
ax.plot(resampled_left['tracker_pos'][2], label='x tracker tr 3')
ax.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1)
ax = fig.add_subplot(1, 2, 2)
ax.set_ylim(lims)
ax.plot(resampled_left['target_pos'][3], label='x target')
ax.plot(resampled_left['tracker_pos'][3], label='x tracker tr 4')
ax.plot(resampled_left['tracker_pos'][4], label='x tracker tr 5')
ax.plot(resampled_left['tracker_pos'][5], label='x tracker tr 6')
ax.legend(loc="lower right", fontsize="small", markerscale=0.5, labelspacing=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'left_agent.eps')


fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 2, 1)
ax.set_ylim(lims)
ax.plot(resampled_right['target_pos'][0], label='x target')
ax.plot(resampled_right['tracker_pos'][0], label='x tracker tr 1')
ax.plot(resampled_right['tracker_pos'][1], label='x tracker tr 2')
ax.plot(resampled_right['tracker_pos'][2], label='x tracker tr 3')
ax.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1)
ax = fig.add_subplot(1, 2, 2)
ax.set_ylim(lims)
ax.plot(resampled_right['target_pos'][3], label='x target')
ax.plot(resampled_right['tracker_pos'][3], label='x tracker tr 4')
ax.plot(resampled_right['tracker_pos'][4], label='x tracker tr 5')
ax.plot(resampled_right['tracker_pos'][5], label='x tracker tr 6')
ax.legend(loc="lower right", fontsize="small", markerscale=0.5, labelspacing=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'right_agent.eps')


"""Playback experiment"""
td_left = az.run_agent_with_playback(agent_directory, last_gen, 'left')
td_right = az.run_agent_with_playback(agent_directory, last_gen, 'right')
resampled_left = resample_behavior(td_left)
resampled_right = resample_behavior(td_right)


fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 2, 1)
ax.set_ylim(lims)
ax.plot(resampled_left['target_pos'][0], label='x target')
ax.plot(resampled_left['tracker_pos'][0], label='x tracker tr 1')
ax.plot(resampled_left['tracker_pos'][1], label='x tracker tr 2')
ax.plot(resampled_left['tracker_pos'][2], label='x tracker tr 3')
ax.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1)
ax = fig.add_subplot(1, 2, 2)
ax.set_ylim(lims)
ax.plot(resampled_left['target_pos'][3], label='x target')
ax.plot(resampled_left['tracker_pos'][3], label='x tracker tr 4')
ax.plot(resampled_left['tracker_pos'][4], label='x tracker tr 5')
ax.plot(resampled_left['tracker_pos'][5], label='x tracker tr 6')
ax.legend(loc="lower right", fontsize="small", markerscale=0.5, labelspacing=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'left_agent_playback.eps')


fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 2, 1)
ax.set_ylim(lims)
ax.plot(resampled_right['target_pos'][0], label='x target')
ax.plot(resampled_right['tracker_pos'][0], label='x tracker tr 1')
ax.plot(resampled_right['tracker_pos'][1], label='x tracker tr 2')
ax.plot(resampled_right['tracker_pos'][2], label='x tracker tr 3')
ax.legend(loc="upper right", fontsize="small", markerscale=0.5, labelspacing=0.1)
ax = fig.add_subplot(1, 2, 2)
ax.set_ylim(lims)
ax.plot(resampled_right['target_pos'][3], label='x target')
ax.plot(resampled_right['tracker_pos'][3], label='x tracker tr 4')
ax.plot(resampled_right['tracker_pos'][4], label='x tracker tr 5')
ax.plot(resampled_right['tracker_pos'][5], label='x tracker tr 6')
ax.legend(loc="lower right", fontsize="small", markerscale=0.5, labelspacing=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'right_agent_playback.eps')


"""
Generalization:
- other initial target location
- other target speeds
- other number of target turns
- other boundary size
"""

td_gen = az.check_joint_generalization(agent_directory, ag1, ag2)
resampled_startpos = resample_trials(td_gen['startpos'])
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)
for trial in range(len(resampled_startpos['target_pos'])):
    ax.set_ylim(lims)
    ax.plot(resampled_startpos['target_pos'][trial] - resampled_startpos['tracker_pos'][trial])
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'startpos_gen.eps')

speed_gen = td_gen['speed']
fig = plt.figure(figsize=(10, 8))
for p in range(4):
    ax = fig.add_subplot(2, 2, p+1)
    ax.set_ylim(lims)
    ax.plot(speed_gen['target_pos'][p], label='x target')
    ax.plot(speed_gen['tracker_pos'][p], label='x tracker')
    ax.plot(speed_gen['tracker_v'][p], label='v tracker')
    ax.plot(speed_gen['keypress'][p][:, 0], label='v left')
    ax.plot(speed_gen['keypress'][p][:, 1], label='v right')

ax.legend(loc="lower right", fontsize="small", markerscale=0.5, labelspacing=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'velocity_gen.eps')

# az.plot_data(td_gen['turns'], "all", "Trial behavior", lims)

fig = plt.figure(figsize=(15, 8))
for p in range(num_trials):
    ax = fig.add_subplot(2, 3, p+1)
    ax.set_ylim(-30, 30)
    ax.plot(td_gen['width']['target_pos'][p], label='x target')
    ax.plot(td_gen['width']['tracker_pos'][p], label='x tracker')
    ax.plot(td_gen['width']['tracker_v'][p], label='v tracker')
    ax.plot(td_gen['width']['keypress'][p][:, 0], label='v left')
    ax.plot(td_gen['width']['keypress'][p][:, 1], label='v right')

ax.legend(loc="lower right", fontsize="small", markerscale=0.5, labelspacing=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'width_gen.eps')


meanf = (np.mean(td_gen['startpos']['fitness']) + np.mean(td_gen['speed']['fitness']) +
         np.mean(td_gen['turns']['fitness']) + np.mean(td_gen['width']['fitness'])) / 4
print(meanf)

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

az.plot_data(td_deaf_start, "all", "Trial behavior", lims)
az.plot_data(td_deaf_before_half, "all", "Trial behavior", lims)
az.plot_data(td_deaf_half, "all", "Trial behavior", lims)


td_borderblind_start = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "start",
                                                    "visual_border", savedata=True)
td_borderblind_before_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "before_midturn",
                                                          "visual_border", savedata=True)
td_borderblind_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "midturn",
                                                   "visual_border", savedata=True)
az.plot_data(td_borderblind_start, "all", "Trial behavior", lims)
az.plot_data(td_borderblind_before_half, "all", "Trial behavior", lims)
az.plot_data(td_borderblind_half, "all", "Trial behavior", lims)


td_targetblind_start = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "start",
                                                    "visual_target", savedata=True)
td_targetblind_before_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "before_midturn",
                                                          "visual_target", savedata=True)
td_targetblind_half = lesion_test.run_joint_trials(ag1, ag2, lesion_test.trials, "midturn",
                                                   "visual_target", savedata=True)
az.plot_data(td_targetblind_start, "all", "Trial behavior", lims)
az.plot_data(td_targetblind_before_half, "all", "Trial behavior", lims)
az.plot_data(td_targetblind_half, "all", "Trial behavior", lims)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(3, 2, 1)
ax.plot(td_gen['speed']['target_pos'][0], label='x target')
ax.plot(td_gen['speed']['tracker_pos'][0], label='x tracker')
ax.set_title("A. Faster target", fontsize=16)
ax.legend(loc="upper right", fontsize="medium", markerscale=0.5, labelspacing=0.1)
sns.despine()

ax = fig.add_subplot(3, 2, 2)
ax.plot(td_deaf_start['target_pos'][0], label='Target position')
ax.plot(td_deaf_start['tracker_pos'][0], label='Tracker position')
ax.set_title("B. Deaf agents", fontsize=16)
sns.despine()

ax = fig.add_subplot(3, 2, 3)
ax.plot(td_gen['speed']['target_pos'][1], label='Target position')
ax.plot(td_gen['speed']['tracker_pos'][1], label='Tracker position')
ax.set_title("C. Slower target", fontsize=16)
sns.despine()

ax = fig.add_subplot(3, 2, 4)
ax.plot(td_borderblind_start['target_pos'][0], label='x target')
ax.plot(td_borderblind_start['tracker_pos'][0], label='x tracker')
ax.set_title("D. Border-blind agents", fontsize=16)
sns.despine()

ax = fig.add_subplot(3, 2, 5)
ax.plot(td_gen['width']['target_pos'][4], label='x target')
ax.plot(td_gen['width']['tracker_pos'][4], label='x tracker')
ax.set_title("E. Widened environment", fontsize=16)
sns.despine()

ax = fig.add_subplot(3, 2, 6)
ax.plot(td_targetblind_start['target_pos'][0], label='x target')
ax.plot(td_targetblind_start['tracker_pos'][0], label='x tracker')
ax.set_title("F. Target-blind agents", fontsize=16)
sns.despine()

plt.tight_layout()
plt.savefig(figure_directory + 'generalizations.eps')



"""
Behavioral dynamics
- steady-state velocity for different static tracker/target position combinations
- overall for tracker velocity
- separately for left and right agent
"""

td_steady = az.run_best_pair_steady(agent_directory, ag1, ag2)

positions = list(range(-20, 20))
# Define the sample space (plotting ranges)
ymin = min(positions)
ymax = max(positions)
num_points = len(positions)
# Define plotting grid
y1 = np.linspace(ymin, ymax, num_points)
y2 = np.linspace(ymin, ymax, num_points)
Y1, Y2 = np.meshgrid(y1, y2)


def get_vel_dynamics(which_vel):
    # overall vel
    changes_y1 = np.zeros((num_points, num_points))
    # fetch the velocities
    for i in range(len(positions)):
        for j in range(len(positions)):
            if which_vel == 'combined':
                changes_y1[i, j] = td_steady[positions[i]]['tracker_v'][j][-1]
            elif which_vel == 'left':
                changes_y1[i, j] = td_steady[positions[i]]['keypress'][j][-1, 0]
            elif which_vel == 'right':
                changes_y1[i, j] = td_steady[positions[i]]['keypress'][j][-1, 1]
    changes_y2 = np.zeros((num_points, num_points))

    return changes_y1, changes_y2


# plot the phase portrait
tr_vec = get_vel_dynamics('combined')
plt.figure(figsize=(5, 4))
plt.quiver(Y1, Y2, tr_vec[0], tr_vec[1], color='b', alpha=.75)
plt.box('off')
plt.xlabel('x target')
plt.ylabel('x tracker')
plt.title('Steady-state tracker velocity field', fontsize=16)
# plt.show()
plt.savefig(figure_directory + 'velgrid.eps')


left_vec = get_vel_dynamics('left')
right_vec = get_vel_dynamics('right')

# 2 strategies comparison
fig = plt.figure(figsize=(10, 4))

ax = fig.add_subplot(1, 2, 1)
ax.quiver(Y1, Y2, left_vec[0], left_vec[1], color='r', alpha=.75)
ax.set_xlabel('x target')
ax.set_ylabel('x tracker')

ax = fig.add_subplot(1, 2, 2)
ax.quiver(Y1, Y2, right_vec[0], right_vec[1], color='g', alpha=.75)
ax.set_xlabel('x target')
ax.set_ylabel('x tracker')
sns.despine()

plt.tight_layout()
plt.savefig(figure_directory + 'velgrid_motors.eps')


"""
Perception input-output
"""

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(2, 2, 1)
ax.set_title("A. Border sensors input vs neuronal output", fontsize=16)
input_left_border = td['input_a1'][5][:, 0] / ag1.VW[0] / 10
input_right_border = td['input_a1'][5][:, 1] / ag1.VW[1] / 10
ax.plot(input_left_border, alpha=0.5)
ax.plot(input_right_border, alpha=0.5)
ax.plot(td['output_a1'][5][:, 0], label='ag1 n1')
ax.plot(td['output_a1'][5][:, 1], label='ag1 n2')
ax.plot(td['output_a2'][5][:, 0], label='ag2 n1')
ax.plot(td['output_a2'][5][:, 1], label='ag2 n2')
ax.legend(loc="best", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)

ax = fig.add_subplot(2, 2, 2)
ax.set_title("B. Target sensors input vs neuronal output", fontsize=16)
input_left_target = td['input_a1'][5][:, 2] / ag1.VW[2] / 10
input_right_target = td['input_a1'][5][:, 3] / ag1.VW[3] / 10
ax.plot(input_left_target, alpha=0.5)
ax.plot(input_right_target, alpha=0.5)
ax.plot(td['output_a1'][5][:, 2], label='ag1 n3')
ax.plot(td['output_a1'][5][:, 3], label='ag1 n4')
ax.legend(loc="best", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)

ax = fig.add_subplot(2, 2, 3)
ax.set_title("C. Target sensors input vs neuronal output", fontsize=16)
input_left_target = td['input_a2'][5][:, 2] / ag2.VW[2] / 10
input_right_target = td['input_a2'][5][:, 3] / ag2.VW[3] / 10
ax.plot(input_left_target, alpha=0.5)
ax.plot(input_right_target, alpha=0.5)
ax.plot(td['output_a2'][5][:, 2], label='ag2 n3')
ax.plot(td['output_a2'][5][:, 3], label='ag2 n4')
ax.legend(loc="best", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)

ax = fig.add_subplot(2, 2, 4)
ax.set_title("D. Audio sensors input vs neuronal output", fontsize=16)
input_left_audio = td['input_a1'][5][:, 4] / ag1.AW[0] / 10
input_right_audio = td['input_a1'][5][:, 5] / ag1.AW[1] / 10
ax.plot(input_left_audio, label='left eye/ear', alpha=0.5)
ax.plot(input_right_audio, label='right eye/ear', alpha=0.5)
ax.plot(td['output_a1'][5][:, 4], label='ag1 n5')
ax.plot(td['output_a1'][5][:, 5], label='ag1 n6')
ax.plot(td['output_a2'][5][:, 4], label='ag2 n5')
ax.plot(td['output_a2'][5][:, 5], label='ag2 n6')
ax.legend(loc="center left", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)

sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'inputs.eps')


fig = plt.figure(figsize=(15, 8))

for p in range(num_trials):
    ax = fig.add_subplot(2, 3, p+1)
    ax.set_ylim(0, 1)
    for i in range(8):
        ax.plot(resampled_td['output_a1'][p][:, i], label='n{}'.format(i+1))

ax.legend(loc="best", fontsize="medium", markerscale=0.5, labelspacing=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'ag1_outputs.eps')

fig = plt.figure(figsize=(15, 8))

for p in range(num_trials):
    ax = fig.add_subplot(2, 3, p+1)
    ax.set_ylim(0, 1)
    for i in range(8):
        ax.plot(resampled_td['output_a2'][p][:, i], label='n{}'.format(i+1))

ax.legend(loc="best", fontsize="medium", markerscale=0.5, labelspacing=0.1)
sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'ag2_outputs.eps')


"""
Motor input-output
"""

fig = plt.figure(figsize=(6, 4))
fig.suptitle("Motor neurons output vs motor activation", fontsize=16)
plt.plot(td['output_a1'][1][:, 6], label="ag1 n7")
plt.plot(td['output_a1'][1][:, 7], label="ag1 n8")
scaled_left = td['keypress'][1][:, 0] / 10
plt.plot(scaled_left, label="left motor")
plt.plot(td['output_a2'][1][:, 6], label="ag2 n7")
plt.plot(td['output_a2'][1][:, 7], label="ag2 n8")
scaled_right = td['keypress'][1][:, 1] / 10
plt.plot(scaled_right, label="right motor")
plt.legend(loc="lower left", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)
fig.savefig(figure_directory + "motors.eps")


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


""" AIS """

pkl_file = open('ais_trial2.pkl', 'rb')
ais2 = pickle.load(pkl_file)
pkl_file.close()


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(2, 2, 1)
for i in range(4):
    ax.plot(list(smooth(np.array(ais2[1]['activation_a1'][i]), 25, 'blackman')), label='n{}'.format(i + 1))
ax.set_title('AIS in activation of ag1, n1-4, tr2')
ax = fig.add_subplot(2, 2, 2)
for i in range(4):
    ax.plot(list(smooth(np.array(ais2[1]['activation_a2'][i]), 25, 'blackman')), label='n{}'.format(i + 1))
ax.set_title('AIS in activation of ag2, n1-4, tr2')
plt.legend(loc="lower left", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)
ax = fig.add_subplot(2, 2, 3)
for i in range(4, 8):
    ax.plot(list(smooth(np.array(ais2[1]['activation_a1'][i]), 25, 'blackman')), label='n{}'.format(i + 1))
ax.set_title('AIS in activation of ag1, n5-8, tr2')
ax = fig.add_subplot(2, 2, 4)
for i in range(4, 8):
    ax.plot(list(smooth(np.array(ais2[1]['activation_a2'][i]), 25, 'blackman')), label='n{}'.format(i + 1))
ax.set_title('AIS in activation of ag2, n5-8, tr2')
plt.legend(loc="lower left", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)
plt.tight_layout()
sns.despine()
fig.savefig(figure_directory + 'ais.eps')


""" TE """


pkl_file = open('te_trial2.pkl', 'rb')
te2 = pickle.load(pkl_file)
pkl_file.close()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(2, 2, 1)
for i in range(4):
    ax.plot(list(smooth(np.array(te2[1]['target_dist']['activation_a1'][i]), 25, 'blackman')), label='n{}'.format(i + 1))
ax.set_title('TE D_tg to activation of ag1, n1-4, tr2')
ax = fig.add_subplot(2, 2, 2)
for i in range(4):
    ax.plot(list(smooth(np.array(te2[1]['target_dist']['activation_a2'][i]), 25, 'blackman')), label='n{}'.format(i + 1))
ax.set_title('TE D_tg to activation of ag2, n1-4, tr2')
ax = fig.add_subplot(2, 2, 3)
for i in range(4):
    ax.plot(list(smooth(np.array(te2[1]['left_border_dist']['activation_a1'][i]), 25, 'blackman')), label='n{}'.format(i + 1))
ax.set_title('TE D_bd to activation of ag1, n1-4, tr2')
ax = fig.add_subplot(2, 2, 4)
for i in range(4):
    ax.plot(list(smooth(np.array(te2[1]['left_border_dist']['activation_a2'][i]), 25, 'blackman')), label='n{}'.format(i + 1))
ax.set_title('TE D_bd to activation of ag2, n1-4, tr2')
plt.legend(loc="upper left", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)
plt.tight_layout()
plt.ylim(-0.5, 2)
sns.despine()
fig.savefig(figure_directory + 'te_visual.eps')


pkl_file = open('ais_trial3.pkl', 'rb')
ais3 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('te_trial3.pkl', 'rb')
te3 = pickle.load(pkl_file)
pkl_file.close()


target_dist = (td['target_pos'][2] - td['tracker_pos'][2]) / 10
border_dist = (20 - td['tracker_pos'][2]) / 20
left_vel = (td['keypress'][2][:, 0]) / 10
right_vel = (td['keypress'][2][:, 1]) / 10

pkl_file = open('te_trial3_exit1.pkl', 'rb')
te_b1_to_left = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('te_trial3_exit2.pkl', 'rb')
te_b2_to_right = pickle.load(pkl_file)
pkl_file.close()

smoothed_a1_n8 = list(smooth(np.array(te_b1_to_left[1]['outputs_a1_n7'][0]), 25, 'blackman'))
to_plot_a1_n8 = [num if num > 0 else 0 for num in smoothed_a1_n8]

smoothed_a2_n8 = list(smooth(np.array(te_b2_to_right[1]['outputs_a2_n7'][0]), 25, 'blackman'))
to_plot_a2_n7 = [num if num > 0 else 0 for num in smoothed_a2_n8]


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(2, 2, 1)
ax.plot(list(smooth(np.array(ais3[1]['outputs_a1'][3]), 25, 'blackman')), label='AIS')
ax.plot(list(smooth(np.array(te3[1]['target_dist']['outputs_a1'][3]), 25, 'blackman')),
         label='TE D_tg')
ax.plot(target_dist, label='D_tg')
ax.axhline(0, color='black')
ax.set_title('A. Information dynamics ag1, n4', fontsize=16)

ax = fig.add_subplot(2, 2, 2)
ax.plot(list(smooth(np.array(ais3[1]['outputs_a1'][6]), 25, 'blackman')), label='AIS')
ax.plot(list(smooth(np.array(te3[1]['target_dist']['outputs_a1'][6]), 25, 'blackman')),
         label='TE D_tg')
ax.plot(target_dist, label='D_tg')
ax.axhline(0, color='black')
ax.set_title('B. Information dynamics ag1, n7', fontsize=16)
ax.legend(loc="center left", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)

ax = fig.add_subplot(2, 2, 3)
ax.plot(to_plot_a1_n8, label='TE to Left motor')
ax.plot(left_vel, label='Left v')
ax.set_title('C. Information transfer ag1, n8', fontsize=16)
ax.legend(loc="best", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)

ax = fig.add_subplot(2, 2, 4)
ax.plot(to_plot_a2_n7, label='TE to Right motor')
ax.plot(right_vel, label='Right v')
ax.axhline(0, color='black')
ax.set_title('D. Information transfer ag2, n7', fontsize=16)
ax.legend(loc="upper left", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)

plt.tight_layout()
sns.despine()
plt.savefig(figure_directory + 'infdyn.eps')


""" Cross MI """

pkl_file = open('cross_mi_trial2.pkl', 'rb')
cross_mi2 = pickle.load(pkl_file)
pkl_file.close()


fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(2, 2, 1)
data = cross_mi2[1][1]['local_mi']['activation_a1']
ax.set_title('L_ag activation n1-4 to R_mot, tr2', fontsize=16)
for i in range(4):
    ax.plot(list(smooth(np.array(data[i]), 25, 'blackman')), label='n{}'.format(i+1))
ax = fig.add_subplot(2, 2, 2)

data = cross_mi2[0][1]['local_mi']['activation_a2']
ax.set_title('R_ag activation n1-4 to L_mot, tr2', fontsize=16)
for i in range(4):
    ax.plot(list(smooth(np.array(data[i]), 25, 'blackman')), label='n{}'.format(i+1))
ax.legend(loc="upper left", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)

ax = fig.add_subplot(2, 2, 3)
data = cross_mi2[1][1]['local_mi']['activation_a1']
ax.set_title('L_ag activation n5-8 to R_mot, tr2', fontsize=16)
for i in range(4, 8):
    ax.plot(list(smooth(np.array(data[i]), 25, 'blackman')), label='n{}'.format(i+1))

ax = fig.add_subplot(2, 2, 4)
data = cross_mi2[0][1]['local_mi']['activation_a2']
ax.set_title('R_ag activation n5-8 to L_mot, tr2', fontsize=16)
for i in range(4, 8):
    ax.plot(list(smooth(np.array(data[i]), 25, 'blackman')), label='n{}'.format(i+1))
ax.legend(loc="upper left", fontsize="medium", markerscale=0.5, labelspacing=0.1, framealpha=1)
sns.despine()
plt.tight_layout()
fig.savefig(figure_directory + 'mi.eps')

"""
Network inference
"""

center_res = pickle.load(open(data_directory + 'results_all_trials_center_smallsample.p', 'rb'))
pre_border_res = pickle.load(open(data_directory + 'results_all_trials_pre_border_2rep.p', 'rb'))
center_res21 = pickle.load(open(data_directory + 'network_center_b2m1.p', 'rb'))
pre_border_res21 = pickle.load(open(data_directory + 'network_preborder_b2m1.p', 'rb'))

graph1 = vg.generate_network_graph(center_res, n_nodes=7, fdr=True, find_u='max_te')
graph2 = vg.generate_network_graph(pre_border_res, n_nodes=7, fdr=True, find_u='max_te')
graph3 = vg.generate_network_graph(center_res21, n_nodes=7, fdr=True, find_u='max_te')
graph4 = vg.generate_network_graph(pre_border_res21, n_nodes=7, fdr=True, find_u='max_te')


node_labels = {0: 'x_tg', 1: 'x_tr', 2: 'n5', 3: 'n6', 4: 'n7', 5: 'n8', 6: 'motor'}
pos = nx.circular_layout(graph1)

plt.figure(figsize=(10, 8))
# Plot graph.
ax1 = plt.subplot(221)
nx.draw_circular(graph1, with_labels=False, node_size=1000, alpha=1.0, ax=ax1,
                 node_color='Gainsboro', hold=True, font_size=14,
                 font_weight='bold')
nx.draw_networkx_labels(graph1, pos, labels=node_labels)
edge_labels = nx.get_edge_attributes(graph1, 'weight')
nx.draw_networkx_edge_labels(graph1, pos, edge_labels=edge_labels, font_size=13)
ax1.set_title('A. Center network L_ag to R_motor', fontsize=16)

ax2 = plt.subplot(222)
nx.draw_circular(graph2, with_labels=False, node_size=1000, alpha=1.0, ax=ax2,
                 node_color='Gainsboro', hold=True, font_size=14,
                 font_weight='bold')
nx.draw_networkx_labels(graph2, pos, labels=node_labels)
edge_labels = nx.get_edge_attributes(graph2, 'weight')
nx.draw_networkx_edge_labels(graph2, pos, edge_labels=edge_labels, font_size=13)
ax2.set_title('B. Pre-border network L_ag to R_motor', fontsize=16)

ax3 = plt.subplot(223)
nx.draw_circular(graph3, with_labels=False, node_size=1000, alpha=1.0, ax=ax3,
                 node_color='Gainsboro', hold=True, font_size=14,
                 font_weight='bold')
nx.draw_networkx_labels(graph3, pos, labels=node_labels)
edge_labels = nx.get_edge_attributes(graph3, 'weight')
nx.draw_networkx_edge_labels(graph3, pos, edge_labels=edge_labels, font_size=13)
ax3.set_title('C. Center network R_ag to L_motor', fontsize=16)

ax4 = plt.subplot(224)
nx.draw_circular(graph4, with_labels=False, node_size=1000, alpha=1.0, ax=ax4,
                 node_color='Gainsboro', hold=True, font_size=14,
                 font_weight='bold')
nx.draw_networkx_labels(graph4, pos, labels=node_labels)
edge_labels = nx.get_edge_attributes(graph4, 'weight')
nx.draw_networkx_edge_labels(graph4, pos, edge_labels=edge_labels, font_size=13)
ax4.set_title('D. Pre-border network R_ag to L_motor', fontsize=16)

plt.tight_layout()
plt.savefig(figure_directory + 'networks_compare.eps')
