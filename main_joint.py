"""
This is the main file for running evolution of neural network agents in the Knoblich and Jordan (2003) task.
This version does not parallelize the seeds and can be combined with parallel agent processing.
"""
from dotenv import load_dotenv
import random
from python_simulator.evolve import Evolution, JointEvolution
# from cython_simulator.evolve import JointEvolution
import json
import argparse
import os
import shutil
# from profilestats import profile
# @profile(print_stats=10, dump_stats=True)


load_dotenv()


def main(condition, agent_type, seed_num):
    # load configuration settings
    json_data = open('config.json')
    config = json.load(json_data)
    json_data.close()

    parent_dir = os.path.join(os.getenv("DATA_DIR"), condition, agent_type)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # set random seed
    random.seed(seed_num)

    config['evaluation_params']['velocity_control'] = agent_type
    if agent_type == "direct":
        config['agent_params']["n_visual_sensors"] = 4
        config['agent_params']["n_visual_connections"] = 1
        config['agent_params']["n_audio_connections"] = 1
        config['agent_params']["n_effector_connections"] = 2

    # if mutation_variance:
    #     config['evolution_params']['mutation_variance'] = mutation_variance
    # if prob_crossover:
    #     config['evolution_params']['prob_crossover'] = prob_crossover

    # set up evolution
    if condition == "joint":
        evolution = JointEvolution(config['evolution_params']['pop_size'],
                                   config['evolution_params'],
                                   config['network_params'],
                                   config['evaluation_params'],
                                   config['agent_params'])
    else:  # condition == "single"
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
    # evolution.run_joint(None, parallel_agents=False)
    evolution.run_joint(None, parallel_agents=True)
    # evolution.run_joint(parent_dir + '/seedpop', parallel_agents=True)

    return


if __name__ == '__main__':
    # run with  python simulate.py real > kennylog.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", type=str, help="specify the experimental condition",
                        choices=["joint", "single"])
    parser.add_argument("agent_type", type=str, help="specify the type of the agent you want to run",
                        choices=["buttons", "direct"])
    parser.add_argument("seed_num", type=int, help="specify random seed number")
    # parser.add_argument("-m", "--mutation_variance", type=int, default=1, help="specify the mutation variance")
    # parser.add_argument("-c", "--prob_crossover", type=int, default=0.8, help="specify the probability of crossover")
    args = parser.parse_args()
    main(args.condition, args.agent_type, args.seed_num)
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
