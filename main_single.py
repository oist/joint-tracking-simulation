"""
This is the main file for running evolution of neural network agents in the Knoblich and Jordan (2003) task.
This version does not parallelize the seeds and can be combined with parallel agent processing.
"""
import random
from python_simulator.evolve import Evolution
# from cython_simulator.evolve import Evolution
import json
import argparse
import os
import shutil
# from profilestats import profile
# @profile(print_stats=10, dump_stats=True)


def main(agent_type, seed_num):
    # load configuration settings
    json_data = open('config.json')
    config = json.load(json_data)
    json_data.close()

    parent_dir = os.getcwd() + '/agents/single/' + agent_type
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # set random seed
    random.seed(seed_num)

    config['evaluation_params']['velocity_control'] = agent_type
    if agent_type == "direct":
        config['agent_params']["n_visual_sensors"] = 4
        config['agent_params']["n_visual_connections"] = 1
        config['agent_params']["n_audio_connections"] = 1
        # config['agent_params']["n_effector_connections"] = 1
        config['agent_params']["n_effector_connections"] = 2

    # set up evolution
    evolution = Evolution(config['evolution_params']['pop_size'],
                          config['evolution_params'],
                          config['network_params'],
                          config['evaluation_params'],
                          config['agent_params'])

    # create the right directory
    foldername = parent_dir + '/' + str(seed_num)
    evolution.set_foldername(foldername)
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)

    with open(foldername + '/usedconfig.json', 'w') as fp:
        json.dump(config, fp)

    # run evolution from scratch or starting from a given population
    evolution.run(None, parallel_agents=True)
    # evolution.run(None, parallel_agents=False)
    # evolution.run(150)
    return


if __name__ == '__main__':
    # run with  python simulate.py real > kennylog.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_type", type=str, help="specify the type of the agent you want to run",
                        choices=["buttons", "direct"])
    parser.add_argument("seed_num", type=int, help="specify random seed number")
    args = parser.parse_args()
    main(args.agent_type, args.seed_num)

    # from random import randint
    # print(randint(0, 9))

#     # To parallelize the seeds instead of agents:
#     parser.add_argument("seed_list", nargs='+', type=int)  # seed_num is a list
#     procs = []
#     for seed_num in args.seed_list:
#         proc = Process(target=main, args=(args.agent_type, seed_num))
#         procs.append(proc)
#         proc.start()
#     for proc in procs:
#         proc.join()
