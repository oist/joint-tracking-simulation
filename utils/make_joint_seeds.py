import os
import pickle
from python_simulator.agents import EmbodiedAgentV2, DirectVelocityAgent
from python_simulator.CTRNN import BrainCTRNN
from analyzer.analyze import load_population, load_config


# def get_best_agents(agent_type):
#     agent_directory = "./Agents/single/" + agent_type
#     agent_folders = list(filter(lambda f: not f.startswith('.'), os.listdir(agent_directory)))
#     agents = []
#     for f in agent_folders:
#         seed_files = list(filter(lambda f: f.startswith('gen'),
#                                  os.listdir(agent_directory + '/{}'.format(f))))
#         gen_numbers = [int(x[3:]) for x in seed_files]
#         agents.extend(load_population('single', agent_type, f, max(gen_numbers)))
#     agents.sort(key=lambda agent: agent.fitness, reverse=True)
#     best_agents = agents[:50]
#     return best_agents
#
#
# json_data = open('config.json')
# config = json.load(json_data)
# json_data.close()
#
# bb = get_best_agents('buttons')
# bd = get_best_agents('direct')
#
#
# def export_best_agents(best_agents, agent_type):
#     n_agents = len(best_agents)
#     n_genes = len(best_agents[0].genotype)
#     all_genotypes = np.zeros([n_agents, n_genes])
#     for i in range(len(best_agents)):
#         all_genotypes[i,:] = best_agents[i].genotype
#     np.savetxt('./Agents/joint/{}/seedpop.csv'.format(agent_type), all_genotypes, delimiter=",")
#
#
# export_best_agents(bb, 'buttons')
# export_best_agents(bd, 'direct')


# def reconstruct_agent(agent, agent_type):
#     agent_brain = BrainCTRNN(config['network_params']['num_neurons'],
#                              config['network_params']['step_size'],
#                              config['network_params']['tau_range'],
#                              config['network_params']['g_range'],
#                              config['network_params']['theta_range'],
#                              config['network_params']['w_range'])
#     if agent_type == "buttons":
#         new_agent = EmbodiedAgentV2(agent_brain, config['agent_params'],
#                                 config['evaluation_params']['screen_width'])
#     else:
#         new_agent = DirectVelocityAgent(agent_brain, config['agent_params'],
#                                 config['evaluation_params']['screen_width'])
#     new_agent.__dict__ = deepcopy(agent.__dict__)
#     return new_agent
#
#
# def save_best_agents(agent_type):
#     best_agents = get_best_agents(agent_type)
#     population = []
#     for agent in best_agents:
#         population.append(reconstruct_agent(agent, agent_type))
#     pop_file = open('./Agents/joint/{}/seedpop'.format(agent_type), 'wb')
#     pickle.dump(population, pop_file)
#     pop_file.close()
#
# save_best_agents("buttons")
# save_best_agents("direct")


config = load_config(None)


# def get_best_agents(agent_type):
#     agent_directory = "./Agents/single/" + agent_type
#     agent_folders = list(filter(lambda f: not f.startswith('.'), os.listdir(agent_directory)))
#     good_agents = []
#     for folder in agent_folders:
#         seed_files = list(filter(lambda genfile: genfile.startswith('gen'),
#                                  os.listdir(agent_directory + '/{}'.format(folder))))
#         gen_numbers = [int(x[3:]) for x in seed_files]
#         population = load_population('single', agent_type, folder, max(gen_numbers))
#         # choose only from seeds in which fitness reached at least 90%
#         if population[0].fitness > 0.9:
#             good_agents.extend(population[:10])
#
#     # add only unique genotypes
#     best_agents = []
#     for a in good_agents:
#         if a not in best_agents:
#             best_agents.append(a)
#
#     return best_agents


def get_best_agents(agent_type, seeds):
    good_agents = []
    for seed in seeds:
        agent_directory = "agents/single/" + agent_type + "/" + seed
        seed_files = list(filter(lambda genfile: genfile.startswith('gen'),
                                 os.listdir(agent_directory)))
        gen_numbers = [int(x[3:]) for x in seed_files]
        population = load_population('single', agent_type, seed, max(gen_numbers))
        # choose only from seeds in which fitness reached at least 90%
        good_agents.extend(population[:25])

    # add only unique genotypes
    best_agents = []
    for a in good_agents:
        if a not in best_agents:
            best_agents.append(a)

    return best_agents


# best_buttons = get_best_agents('buttons')
# best_direct = get_best_agents('direct')
best_direct = get_best_agents('direct', ['102575', '328651', '453756'])
best_direct.sort(key=lambda agent: agent.fitness, reverse=True)
best_direct = best_direct[:50]


def create_random_pop(size, agent_type):
    population = []
    for i in range(size):
        # create the agent's CTRNN brain
        agent_brain = BrainCTRNN(config['network_params']['num_neurons'],
                                 config['network_params']['step_size'],
                                 config['network_params']['tau_range'],
                                 config['network_params']['g_range'],
                                 config['network_params']['theta_range'],
                                 config['network_params']['w_range'])

        if agent_type == "direct":
            config['agent_params']["n_visual_sensors"] = 4
            config['agent_params']["n_visual_connections"] = 1
            config['agent_params']["n_audio_connections"] = 1
            config['agent_params']["n_effector_connections"] = 2
            agent = DirectVelocityAgent(agent_brain, config['agent_params'],
                                               config['evaluation_params']['screen_width'])
        else:
            agent = EmbodiedAgentV2(agent_brain, config['agent_params'],
                                           config['evaluation_params']['screen_width'])
        population.append(agent)
    return population


# fillsize = 50 - len(best_direct)
# random_fill = create_random_pop(fillsize, 'direct')
# best_direct.extend(random_fill)


def save_pop(population, agent_type):
    pop_file = open('agents/joint/{}/seedpop'.format(agent_type), 'wb')
    pickle.dump(population, pop_file)
    pop_file.close()


# save_pop(best_buttons, 'buttons')
save_pop(best_direct, 'direct')
