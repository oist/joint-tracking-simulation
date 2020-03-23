from . import analyze as az
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from scipy import signal
import pickle


agent_directory = 'agents'
figure_directory = 'figures/'
last_gen = 2000
lims = (-20, 20)

config = az.load_config(agent_directory)
td, ag1, ag2 = az.run_best_pair(agent_directory, last_gen)


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

output = open('resampled_td_914463.pkl', 'wb')
pickle.dump(resampled_td, output)
output.close()


def plot_trials(trial_data, figure_name):
    # plot just trial behavior
    num_trials = 6
    fig = plt.figure(figsize=(15, 8))

    for p in range(num_trials):
        ax = fig.add_subplot(2, 3, p+1)
        ax.set_ylim(lims)
        ax.plot(trial_data['target_pos'][p], label='x target')
        ax.plot(trial_data['tracker_pos'][p], label='x tracker')
        ax.plot(trial_data['tracker_v'][p], label='v tracker')
        ax.plot(trial_data['keypress'][p][:, 0], label='v left')
        ax.plot(trial_data['keypress'][p][:, 1], label='v right')

    ax.legend(loc="lower right", fontsize="small", markerscale=0.5, labelspacing=0.1)
    sns.despine()
    plt.tight_layout()
    plt.savefig(figure_directory + figure_name)


plot_trials(resampled_td, 'trial_behavior.eps')

"""Playback experiment"""


def resample_playback_trials(trial_data):
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

        sampled_td['keypress'][trial_num] = np.zeros((3100, 2))
        sampled_td['button_state_a1'][trial_num] = np.zeros((3100, 2))

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

        for i in range(2):
            sampled_td['keypress'][trial_num][:, i] = np.concatenate(
                (trial_data['keypress'][trial_num][:100, i],
                 signal.resample(trial_data['keypress'][trial_num][100:, i], 3000)))
            sampled_td['button_state_a1'][trial_num][:, i] = np.concatenate(
                (trial_data['button_state_a1'][trial_num][:100, i],
                 signal.resample(trial_data['button_state_a1'][trial_num][100:, i], 3000)))

    return sampled_td


td_left = az.run_agent_with_playback(agent_directory, last_gen, 'left')
td_right = az.run_agent_with_playback(agent_directory, last_gen, 'right')
resampled_left = resample_playback_trials(td_left)
resampled_right = resample_playback_trials(td_right)

plot_trials(resampled_left, 'left_agent_playback.eps')
plot_trials(resampled_right, 'right_agent_playback.eps')
