from analyzer import analyze as az
import pstats, cProfile

import pyximport
pyximport.install()

from python_simulator import evolve
from python_simulator import simulate


from main_joint import main
cProfile.runctx("main('direct', 123, False, False)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()


#
# config = az.load_config(None, None, None)
# evolution = evolve.Evolution(config['evolution_params']['pop_size'],
#                              config['evolution_params'],
#                              config['network_params'],
#                              config['evaluation_params'],
#                              config['agent_params'])
# pop1, pop2 = evolution.load_joint_population(4)

# tested_pairs = list(zip(pop1, pop2))
# for pair in tested_pairs:
#     evolution.process_pair(pair, 0)


# 10 min per generation of just experiment
# simulation_run = simulate.Simulation(evolution.step_size, evolution.evaluation_params)
# cProfile.runctx("simulation_run.run_joint_trials(pop1[0], pop2[1], simulation_run.trials)", globals(), locals(), "Profile.prof")
# cProfile.runctx("simulation_run.run_trials(pop[0], simulation_run.trials)", globals(), locals(), "Profile.prof")

# 0.007 s to reproduce 2 agents
#cProfile.runctx("evolution.reproduce(pop)", globals(), locals(), "Profile.prof")

# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()



# from CTRNN import CTRNN
# cbrain = CTRNN(8, 0.01, [1, 100], [1, 1], [-15, 15], [-15, 15])
#
# print(cbrain.get_state())
# # cbrain.euler_step()
##
# cProfile.runctx("cbrain.euler_step()", globals(), locals(), "Profile.prof")
#
# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()
#
# print(cbrain.get_state())


