import pickle
import json
from python_simulator.CTRNN import BrainCTRNN
from python_simulator.agents import EmbodiedAgentV2, DirectVelocityAgent
import numpy as np


def load_joint_genotypes(genfile):
    all_genotypes = np.genfromtxt(genfile, delimiter=',')
    return all_genotypes


button_genes = load_joint_genotypes('./agents/joint/buttons/seedpop.csv')
direct_genes = load_joint_genotypes('./agents/joint/direct/seedpop.csv')

json_data = open('../config.json')
config = json.load(json_data)
json_data.close()


def create_agents(genes, agent_type):
    new_population = []
    for i in range(50):
        agent_brain = BrainCTRNN(config['network_params']['num_neurons'],
                                 config['network_params']['step_size'],
                                 config['network_params']['tau_range'],
                                 config['network_params']['g_range'],
                                 config['network_params']['theta_range'],
                                 config['network_params']['w_range'])

        if agent_type == "buttons":
            new_agent = EmbodiedAgentV2(agent_brain, config['agent_params'],
                                        config['evaluation_params']['screen_width'])
        else:
            new_agent = DirectVelocityAgent(agent_brain, config['agent_params'],
                                            config['evaluation_params']['screen_width'])

        new_agent.genotype = genes[i]
        new_agent.make_params_from_genotype(genes[i])
        new_population.append(new_agent)
    return new_population

buttons_pop = create_agents(button_genes, 'buttons')
direct_pop = create_agents(direct_genes, 'direct')


def save_pop(population, agent_type):
    pop_file = open('./agents/joint/{}/seedpop'.format(agent_type), 'wb')
    pickle.dump(population, pop_file)
    pop_file.close()


save_pop(buttons_pop, 'buttons')
save_pop(direct_pop, 'direct')


# g1 = button_genes[0]
# a1 = buttons_pop[0]
# print(g1 == a1.genotype)
#
# # calculate crossover points
# n_evp = len(config['agent_params']['evolvable_params'])  # how many parameters in addition to weights are evolved
# crossover_points = [i * (n_evp + 8) for i in range(1, 8 + 1)]
# crossover_points.extend([crossover_points[-1] + len(a1.VW),
#                          crossover_points[-1] + len(a1.VW) + len(a1.AW)])
#
#
# def linmap(vin, rin, rout):
#     a = rin[0]
#     b = rin[1]
#     c = rout[0]
#     d = rout[1]
#     return ((c + d) + (d - c) * ((2 * vin - (a + b)) / (b - a))) / 2
#
#
# genorest, vw, aw, mw = np.hsplit(g1, crossover_points[-3:])
# print(linmap(vw, [0, 1], a1.r_range) == a1.VW)
# print(linmap(aw, [0, 1], a1.r_range) == a1.AW)
# print(linmap(mw, [0, 1], a1.e_range) == a1.MW)
#
# unflattened = genorest.reshape(n_evp + 8, 8, order='F')
# tau, theta, w = (np.squeeze(a) for a in np.vsplit(unflattened, [1, 2]))
#
# print(linmap(tau, [0, 1], [1, 100]) == a1.brain.Tau)

# def load_joint_population(genfile):
#     pop_file = open(genfile, 'rb')
#     population = pickle.load(pop_file)
#     pop_file.close()
#     return population
#
# bpop = load_joint_population('./Agents/joint/buttons/seedpop')
# dpop = load_joint_population('./Agents/joint/direct/seedpop')
#
# json_data = open('config.json')
# config = json.load(json_data)
# json_data.close()
#
#
# def convert_agent(agent, agent_type):
#     agent_brain = BrainCTRNN(config['network_params']['num_neurons'],
#                              config['network_params']['step_size'],
#                              config['network_params']['tau_range'],
#                              config['network_params']['g_range'],
#                              config['network_params']['theta_range'],
#                              config['network_params']['w_range'])
#     agent_brain.reassign_attributes(agent.brain.Tau, agent.brain.G, agent.brain.W, agent.brain.Theta)
#
#     if agent_type == "buttons":
#         new_agent = EmbodiedAgentV2(agent_brain, config['agent_params'],
#                                     config['evaluation_params']['screen_width'])
#     else:
#         new_agent = DirectVelocityAgent(agent_brain, config['agent_params'],
#                                         config['evaluation_params']['screen_width'])
#     new_agent.reassign_attributes(agent.VW, agent.AW, agent.MW)
#     return new_agent
#
# new_population = []
#
# for agent in bpop:
#     new_population.append(convert_agent(agent, 'buttons'))
#
# pop_file = open('./Agents/joint/buttons/seedpop2', 'wb')
# pickle.dump(new_population, pop_file)
# pop_file.close()
