import analyze as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


agent_directory = 'agents'
figure_directory = 'figures/'
config = az.load_config(agent_directory)


with open('resampled_td_914463.pkl', 'rb') as pkl:
    resampled_td = pickle.load(pkl)

with open('resampled_td_176176.pkl', 'rb') as pkl:
    resampled_td_176 = pickle.load(pkl)


# behavioral results fig
fig = plt.figure(figsize=(16, 4))

ax = fig.add_subplot(1, 4, 1)
ax.plot(resampled_td_176['target_pos'][1], label='x target')
ax.plot(resampled_td_176['tracker_pos'][1], label='x tracker')
ax.plot(resampled_td_176['tracker_v'][1], label='v tracker')
ax.plot(resampled_td_176['keypress'][1][:, 0], label='v left')
ax.plot(resampled_td_176['keypress'][1][:, 1], label='v right')
ax.set_title("A", fontsize=16, position=(0.1, 0.9))
ax.legend(loc="upper right", fontsize="medium", markerscale=0.5, labelspacing=0.1)

ax = fig.add_subplot(1, 4, 2)
ax.plot(resampled_td['target_pos'][1], label='x target')
ax.plot(resampled_td['tracker_pos'][1], label='x tracker')
ax.plot(resampled_td['tracker_v'][1], label='v tracker')
ax.plot(resampled_td['keypress'][1][:, 0], label='v left')
ax.plot(resampled_td['keypress'][1][:, 1], label='v right')
ax.set_title("B", fontsize=16, position=(0.1, 0.9))

ax = fig.add_subplot(1, 4, 3)
ax.plot(resampled_td['target_pos'][4], label='x target')
ax.plot(resampled_td['tracker_pos'][4], label='x tracker')
ax.plot(resampled_td['tracker_v'][4], label='v tracker')
ax.plot(resampled_td['keypress'][4][:, 0], label='v left')
ax.plot(resampled_td['keypress'][4][:, 1], label='v right')
ax.set_title("C", fontsize=16, position=(0.1, 0.9))

ax = fig.add_subplot(1, 4, 4)
ax.plot(resampled_td['target_pos'][2], label='x target')
ax.plot(resampled_td['tracker_pos'][2], label='x tracker')
ax.plot(resampled_td['tracker_v'][2], label='v tracker')
ax.plot(resampled_td['keypress'][2][:, 0], label='v left')
ax.plot(resampled_td['keypress'][2][:, 1], label='v right')
ax.set_title("D", fontsize=16, position=(0.1, 0.9))

sns.despine()
plt.tight_layout()
plt.savefig(figure_directory + 'alife_strategies.eps')
