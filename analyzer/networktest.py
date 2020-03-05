# Import classes
# install itdxl with: pip install git+git://github.com/pwollstadt/IDTxl.git
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network
import numpy as np
import pickle
from idtxl.network_comparison import NetworkComparison

"""
before border:
400-600
1400-1600
2400-2600

after border:
600-800
1600-1800
2600-2800

around middle:
950-1250
1950-2250
"""


# https://github.com/pwollstadt/IDTxl/wiki/The-Data-Class
# a) Generate test data
# data = Data()
# data.generate_mute_data(n_samples=1000, n_replications=5)

# Real data
pkl_file = open('resampled_td_914463.pkl', 'rb')
td = pickle.load(pkl_file)
pkl_file.close()


# pre_border_ix = np.array([np.arange(400, 600), np.arange(1400, 1600), np.arange(2400, 2600)]).reshape(600)
# after_border_ix = np.array([np.arange(600, 800), np.arange(1600, 1800), np.arange(2600, 2800)]).reshape(600)
# center_ix = np.array([np.arange(950, 1250), np.arange(1950, 2250)]).reshape(600)

pre_border_ix = [range(1350, 1600), range(2350, 2600)]
# after_border_ix = np.array([np.arange(600, 800), np.arange(1600, 1800), np.arange(2600, 2800)]).reshape(600)
center_ix = [range(975, 1225), range(1975, 2225)]


starting = True
for i in range(6):
    for ix in center_ix:
        activations_a1 = td['brain_state_a1'][i][ix, 4:8]
        activations_a2 = td['brain_state_a2'][i][ix, 4:8]
        left_motor = td['keypress'][i][ix, 0].tolist()
        right_motor = td['keypress'][i][ix, 1].tolist()
        target_pos = td['target_pos'][i][ix]
        tracker_pos = td['tracker_pos'][i][ix]
        # b1_to_m2_center = np.column_stack((target_pos, tracker_pos, activations_a1, right_motor))
        b2_to_m1_center = np.column_stack((target_pos, tracker_pos, activations_a2, left_motor))
        if starting:
            # all_trials = b1_to_m2_center
            all_trials = b2_to_m1_center
            starting = False
        else:
            # all_trials = np.dstack((all_trials, b1_to_m2_center))
            all_trials = np.dstack((all_trials, b2_to_m1_center))
data_center = Data(all_trials, dim_order='spr')


starting = True
for i in range(6):
    for ix in pre_border_ix:
        activations_a1 = td['brain_state_a1'][i][ix, 4:8]
        activations_a2 = td['brain_state_a2'][i][ix, 4:8]
        left_motor = td['keypress'][i][ix, 0].tolist()
        right_motor = td['keypress'][i][ix, 1].tolist()
        target_pos = td['target_pos'][i][ix]
        tracker_pos = td['tracker_pos'][i][ix]
        # b1_to_m2_pre_border = np.column_stack((target_pos, tracker_pos, activations_a1, right_motor))
        b2_to_m1_pre_border = np.column_stack((target_pos, tracker_pos, activations_a2, left_motor))
        if starting:
            # all_trials = b1_to_m2_pre_border
            all_trials = b2_to_m1_pre_border
            starting = False
        else:
            # all_trials = np.dstack((all_trials, b1_to_m2_pre_border))
            all_trials = np.dstack((all_trials, b2_to_m1_pre_border))
data_pre_border = Data(all_trials, dim_order='spr')


# all_trials = None
# for i in range(6):
#     activations_a1 = td['brain_state_a1'][i][:, 4:8]
#     activations_a2 = td['brain_state_a2'][i][:, 4:8]
#     left_motor = td['keypress'][i][:, 0].tolist()
#     right_motor = td['keypress'][i][:, 1].tolist()
#     target_pos = td['target_pos'][i]
#     tracker_pos = td['tracker_pos'][i]
#     b1_to_m2 = np.column_stack((target_pos, tracker_pos, activations_a1, right_motor))
#     if i == 0:
#         all_trials = b1_to_m2
#     else:
#         all_trials = np.dstack((all_trials, b1_to_m2))
# data = Data(all_trials, dim_order='spr')


# b) Initialise analysis object and define settings
network_analysis = MultivariateTE()

# estimator choice: for non-linear continuous data use Kraskov:
# JidtKraskovAIS, JidtKraskovCMI, JidtKraskovMI, JidtKraskovTE

# statistical testing - shuffle over time for a max of 200 permutations with standard alpha
# conditioning: (process index, lag)

settings = {'cmi_estimator': 'JidtKraskovCMI',
            'history_target': 1,
            'n_perm_max_stat': 100,
            'alpha_max_stat': 0.05,
            'permute_in_time': True,
            'max_lag_sources': 5,
            'min_lag_sources': 1,
            'add_conditionals': [(0, 1), (0, 2), (1, 1), (1, 2)]}

# c) Run analysis
center_res = network_analysis.analyse_network(settings=settings, data=data_center, targets=[6], sources=[2, 3, 4, 5])
pre_border_res = network_analysis.analyse_network(settings=settings, data=data_pre_border, targets=[6], sources=[2, 3, 4, 5])


# d) Plot inferred network to console and via matplotlib
plot_network(res=center_res, n_nodes=data_center.n_processes)

# print_res_to_console(data=data_pre_border, res=pre_border_res)
# plot_network(res=pre_border_res, n_nodes=data_pre_border.n_processes)

pickle.dump(center_res, open('network_center_b2m1.p', 'wb'))
pickle.dump(pre_border_res, open('network_preborder_b2m1.p', 'wb'))
# pickle.dump(res1, open('results_all_trials.p', 'wb'))
# results = pickle.load(open('results.p', 'rb'))


# compare
# center_res = pickle.load(open('results_all_trials_center_smallsample.p', 'rb'))
# pre_border_res = pickle.load(open('results_all_trials_pre_border_2rep.p', 'rb'))
#
# comparison_settings = {
#     'stats_type': 'independent',
#     'cmi_estimator': 'JidtKraskovCMI',
#     'tail_comp': 'two',
#     'n_perm_comp': 100,
#     'alpha_comp': 0.05,
#     'verbose': True
# }
#
# comparison = NetworkComparison()
#
# comparison_res = comparison.compare_within(data_a=data_center, data_b=data_pre_border,
#                           network_a=center_res, network_b=pre_border_res,
#                           settings=comparison_settings)
#
# print(comparison_res['sign'])
# pickle.dump(comparison_res, open('network_comparison_dict.p', 'wb'))
