import pickle
import numpy as np
import random
import math
from copy import deepcopy
from .CTRNN import BrainCTRNN
from . import simulate
from . import agents
from joblib import Parallel, delayed
# from joblib.pool import has_shareable_memory
np.seterr(over='ignore')


class Evolution:
    def __init__(self, pop_size, evolution_params, network_params, evaluation_params, agent_params):
        """
        Class that executes genetic search.
        :param pop_size: size of the new population
        :param evolution_params: evolution parameters for the simulation
        :param network_params: neural network parameters for the simulation
        :param evaluation_params: parameters used for running trials during the evaluation
        """
        self.pop_size = pop_size
        self.evolution_params = evolution_params
        self.step_size = network_params['step_size']
        self.network_params = network_params
        self.evaluation_params = evaluation_params
        self.agent_params = agent_params
        self.foldername = None

    def set_foldername(self, text):
        self.foldername = text

    def run(self, gen_to_load, parallel_agents):
        """
        Execute a full search run until some condition is reached.
        :param gen_to_load: which generation to load if starting from an existing population
        :param parallel_agents: switch whether processing of agents within generation should be parallel
        :return: the last population in the search
        """
        gen = 0
        # create initial population or load existing one
        if gen_to_load is None:
            population = self.create_population(self.pop_size)
        else:
            population = load_population(self.foldername, gen_to_load)

        # collect average and best fitness
        avg_fitness = [0.0]
        best_fitness = [0.0]
        best_counter = 0

        # initialize a type of simulation
        simulation_setup = simulate.Simulation(self.step_size, self.evaluation_params)

        while gen < self.evolution_params['max_gens'] + 1:
            print(gen)
            # evaluate all agents on the task

            tested_population = []

            if parallel_agents:
                num_cores = self.evolution_params['num_cores']
                slice_size = int(self.pop_size / num_cores)

                for slice_counter in range(slice_size):
                    print(slice_counter)
                    population_slice = population[slice_counter*num_cores:(slice_counter+1)*num_cores]
                    updated_slice = Parallel(n_jobs=num_cores, prefer="threads")(delayed(self.process_agent)(a, simulation_setup)
                                                               for a in population_slice)
                    tested_population.extend(updated_slice)
            else:
                for agent in population:
                    tested_population.extend([self.process_agent(agent, simulation_setup)])

            # log fitness results: average population fitness
            population_avg_fitness = np.mean(pop_fitness(tested_population)).item()
            avg_fitness.append(round(population_avg_fitness, 3))

            # sort agents by fitness from best to worst
            tested_population.sort(key=lambda ag: ag.fitness, reverse=True)
            # log fitness results: best agent fitness
            bf = round(tested_population[0].fitness, 3)
            best_fitness.append(bf)

            # stop the search if fitness hasn't increased in a set number of generations
            if bf == best_fitness[-2]:
                best_counter += 1

                if best_counter > self.evolution_params['evolution_break']:
                    # save the last population
                    save_population(tested_population, self.foldername, gen)
                    # print("Stopped the search at generation {}".format(gen))

                    # save the average and best fitness lists
                    fitness_log = {'average': avg_fitness, 'best': best_fitness}
                    log_fitness(self.foldername, fitness_log)
                    break
            else:
                best_counter = 0

            # save the intermediate or last population and fitness
            if gen % self.evolution_params['check_int'] == 0 or gen == self.evolution_params['max_gens']:
                save_population(tested_population, self.foldername, gen)
                # print("Saved generation {}".format(gen))
                fitness_log = {'average': avg_fitness, 'best': best_fitness}
                log_fitness(self.foldername, fitness_log)

            # reproduce population
            population = self.reproduce(tested_population)
            gen += 1

    @staticmethod
    def process_agent(agent, simulation_setup):
        # for agent in population:
        # initialize a type of simulation
        # simulation_run = simulate.Simulation(self.step_size, self.evaluation_params)
        # trial_data = simulation_run.run_trials(agent, simulation_run.trials)

        # run the trials and return fitness in all trials
        trial_data = simulation_setup.run_trials(agent, simulation_setup.trials)

        # calculate overall fitness
        # agent.fitness = np.mean(trial_data['fitness'])
        agent.fitness = harmonic_mean(trial_data['fitness'])
        # agent.fitness = min(trial_data['fitness'])
        return agent

    # def run_joint(self, gen_to_load, parallel_agents):
    #     """
    #     Execute a full search run until some condition is reached.
    #     :param gen_to_load: which generation to load if starting from an existing population
    #     :param parallel_agents: switch whether processing of agents within generation should be parallel
    #     :return: the last population in the search
    #     """
    #     gen = 0
    #     # create initial population or load existing one
    #     if gen_to_load is None:
    #         population_left, population_right = self.create_joint_population(self.pop_size)
    #     else:
    #         population_left, population_right = self.load_joint_population(self.foldername, gen_to_load)
    #
    #     # collect average and best fitness
    #     avg_fitness = {'left': [0.0], 'right': [0.0], 'combined': [0.0]}
    #     best_fitness = {'left': [0.0], 'right': [0.0], 'combined': [0.0]}
    #     best_counter = 0
    #
    #     # size of the left and right sub-populations
    #     sub_size = int(self.pop_size / 2)
    #     # how many agents from each sub-population to test
    #     tested_size = int(sub_size / self.evolution_params['tested_proportion'])
    #
    #     while gen < self.evolution_params['max_gens'] + 1:
    #         #  print(gen)
    #         idx_left = random.sample(range(sub_size), tested_size)
    #         idx_right = random.sample(range(sub_size), tested_size)
    #         idx_pairs = [(x, y) for x in idx_left for y in idx_right]
    #         n_pairs = len(idx_pairs)
    #
    #         # set fitness of all agents that will participate in the experiment to 0
    #         for idx in idx_left:
    #             population_left[idx].fitness = 0
    #         for idx in idx_right:
    #             population_right[idx].fitness = 0
    #
    #         fitness_dict = dict.fromkeys(idx_pairs)
    #
    #         if parallel_agents:
    #             # evaluate all agents on the task
    #             num_cores = self.evolution_params['num_cores']
    #             for test_counter in range(int(n_pairs/num_cores)):
    #                 # process num_cores pairs at a time updating the dictionary with their trial fitness
    #                 pairs_slice = idx_pairs[test_counter*num_cores:(test_counter+1)*num_cores]
    #                 Parallel(n_jobs=num_cores)(delayed(has_shareable_memory)
    #                                           (self.process_pair(pair, population_left,
    #                                            population_right, fitness_dict))
    #                                            for pair in pairs_slice)
    #         else:
    #             # evaluate all agents on the task
    #             for pair in idx_pairs:
    #                 self.process_pair(pair, population_left, population_right, fitness_dict)
    #
    #         # update fitness value of tested agents, first sum
    #         for key, value in fitness_dict.items():
    #             population_left[key[0]].fitness += value
    #             population_right[key[1]].fitness += value
    #
    #         # then take the mean from all experiments they took part in
    #         for idx in idx_left:
    #             population_left[idx].fitness = population_left[idx].fitness/tested_size
    #         for idx in idx_right:
    #             population_right[idx].fitness = population_right[idx].fitness/tested_size
    #
    #         pair_avg_fitness = np.mean(list(fitness_dict.values())).item()
    #         avg_fitness['combined'].append(round(pair_avg_fitness, 3))
    #         bf_combined = max(fitness_dict.values())
    #         best_fitness['combined'].append(round(bf_combined, 3))
    #
    #         # log fitness results: average population fitness
    #         avgf_left = np.mean([agent.fitness for agent in population_left]).item()
    #         avgf_right = np.mean([agent.fitness for agent in population_right]).item()
    #
    #         avg_fitness['left'].append(round(avgf_left, 3))
    #         avg_fitness['right'].append(round(avgf_right, 3))
    #
    #         # sort agents by fitness from best to worst
    #         population_left.sort(key=lambda ag: ag.fitness, reverse=True)
    #         population_right.sort(key=lambda ag: ag.fitness, reverse=True)
    #
    #         # log fitness results: best agent fitness
    #         bf_left = round(population_left[0].fitness, 3)
    #         bf_right = round(population_right[0].fitness, 3)
    #
    #         best_fitness['left'].append(bf_left)
    #         best_fitness['right'].append(bf_right)
    #
    #         # stop the search if fitness hasn't increased in a set number of generations
    #         if bf_left == best_fitness['left'][-2] and bf_right == best_fitness['right'][-2]:
    #             best_counter += 1
    #
    #             if best_counter > self.evolution_params['evolution_break']:
    #                 # save the last population
    #                 self.save_population({'left': population_left, 'right': population_right},
    #                                      self.foldername, gen)
    #                 # print("Stopped the search at generation {}".format(gen))
    #
    #                 # save the average and best fitness lists
    #                 fitness_log = {'average': avg_fitness, 'best': best_fitness}
    #                 self.log_fitness(self.foldername, fitness_log)
    #                 break
    #         else:
    #             best_counter = 0
    #
    #         # save the intermediate or last population and fitness
    #         if gen % self.evolution_params['check_int'] == 0 or gen == self.evolution_params['max_gens']:
    #             self.save_population({'left': population_left, 'right': population_right},
    #                                  self.foldername, gen)
    #             # print("Saved generation {}".format(gen))
    #             fitness_log = {'average': avg_fitness, 'best': best_fitness}
    #             self.log_fitness(self.foldername, fitness_log)
    #
    #         # reproduce population
    #         population_left = self.reproduce(population_left)
    #         population_right = self.reproduce(population_right)
    #         gen += 1

    def run_joint(self, gen_to_load, parallel_agents):
        """
        Execute a full search run until some condition is reached.
        :param gen_to_load: which generation to load if starting from an existing population
        :param parallel_agents: switch whether processing of agents within generation should be parallel
        :return: the last population in the search
        """
        gen = 0
        # create initial population or load existing one
        if gen_to_load is None:
            population_left, population_right = self.create_joint_population(self.pop_size)
        else:
            # population_left, population_right = self.load_joint_population(self.foldername, gen_to_load)
            population_left, population_right = load_joint_population(gen_to_load)

        # collect average and best fitness
        # with this implementation we don't save particular trial fitnesses
        avg_fitness_left = [0.0]
        avg_fitness_right = [0.0]
        best_fitness_left = [0.0]
        best_fitness_right = [0.0]

        best_counter = 0

        # size of the left and right sub-populations
        sub_size = int(self.pop_size / 2)
        # how many reshuffles to test, e.g. 5
        tested_shuffles = self.evolution_params['tested_proportion']

        # initialize a type of simulation
        simulation_setup = simulate.Simulation(self.step_size, self.evaluation_params)
        # simulation_run = simulate.SimpleSimulation(evolution.step_size, evolution.evaluation_params)

        while gen < self.evolution_params['max_gens'] + 1:

            shuffle_num = 0

            # repeat tested_shuffles times and accumulate average fitness for each agent
            while shuffle_num < tested_shuffles:
                # population_left stays the same, we shuffle the right one
                random.shuffle(population_right)

                # form pairs from two sub-populations (this will be the size of the sub-population)
                pairs_to_test = list(zip(population_left, population_right))
                tested_pairs = []

                # evaluate all agents on the task
                if parallel_agents:
                    num_cores = self.evolution_params['num_cores']
                    # cdef int slice_counter
                    # cdef int slice_size
                    slice_size = int(sub_size/num_cores)
                    for slice_counter in range(slice_size):

                        # process num_cores pairs at a time updating the dictionary with their trial fitness
                        pairs_slice = pairs_to_test[slice_counter*num_cores:(slice_counter+1)*num_cores]
                        # Parallel(n_jobs=num_cores)(delayed(has_shareable_memory)
                        #                            (self.process_pair(pair, shuffle_num))
                        #                            for pair in pairs_slice)
                        updated_slice = Parallel(n_jobs=num_cores, prefer="threads")(delayed(self.process_pair)
                                                                   (pair, shuffle_num, simulation_setup)
                                                                   for pair in pairs_slice)
                        tested_pairs.extend(updated_slice)
                else:
                    for pair in pairs_to_test:
                        tested_pairs.extend(self.process_pair(pair, shuffle_num, simulation_setup))

                population_left, population_right = [list(x) for x in list(zip(*tested_pairs))]
                shuffle_num += 1

            # log fitness results: average population fitness
            avgf_left = np.mean(pop_fitness(population_left)).item()
            avgf_right = np.mean(pop_fitness(population_right)).item()

            avg_fitness_left.append(round(avgf_left, 3))
            avg_fitness_right.append(round(avgf_right, 3))

            # sort agents by fitness from best to worst
            population_left.sort(key=lambda ag: ag.fitness, reverse=True)
            population_right.sort(key=lambda ag: ag.fitness, reverse=True)

            # log fitness results: best agent fitness
            bf_left = round(population_left[0].fitness, 3)
            bf_right = round(population_right[0].fitness, 3)

            best_fitness_left.append(bf_left)
            best_fitness_right.append(bf_right)

            # stop the search if fitness hasn't increased in a set number of generations
            if bf_left == best_fitness_left[-2] and bf_right == best_fitness_right[-2]:
                best_counter += 1

                if best_counter > self.evolution_params['evolution_break']:
                    # save the last population
                    save_population({'left': population_left, 'right': population_right},
                                    self.foldername, gen)
                    # print("Stopped the search at generation {}".format(gen))

                    # save the average and best fitness lists
                    fitness_log = {'average': {'left': avg_fitness_left, 'right': avg_fitness_right},
                                   'best': {'left': best_fitness_left, 'right': best_fitness_right}}
                    log_fitness(self.foldername, fitness_log)
                    break
            else:
                best_counter = 0

            # save the intermediate or last population and fitness
            if gen % self.evolution_params['check_int'] == 0 or gen == self.evolution_params['max_gens']:
                save_population({'left': population_left, 'right': population_right},
                                self.foldername, gen)
                # print("Saved generation {}".format(gen))
                fitness_log = {'average': {'left': avg_fitness_left, 'right': avg_fitness_right},
                               'best': {'left': best_fitness_left, 'right': best_fitness_right}}
                log_fitness(self.foldername, fitness_log)

            # reproduce population
            population_left = self.reproduce(population_left)
            population_right = self.reproduce(population_right)
            gen += 1

    # def process_pair(self, pair, population_left, population_right, fitness_dict):
    #     agent1 = population_left[pair[0]]
    #     agent2 = population_right[pair[1]]
    #
    #     # initialize a type of simulation
    #     simulation_run = simulate.Simulation(self.step_size, self.evaluation_params)
    #     # simulation_run = simulate.SimpleSimulation(evolution.step_size, evolution.evaluation_params)
    #
    #     # run the trials and return fitness in all trials
    #     trial_data = simulation_run.run_joint_trials(agent1, agent2, simulation_run.trials)
    #
    #     # calculate overall fitness for a given trial run
    #     trial_fitness = self.harmonic_mean(trial_data['fitness'])
    #     fitness_dict[pair] = trial_fitness

    @staticmethod
    def process_pair(pair, shuffle_num, simulation_setup):
        agent1 = pair[0]
        agent2 = pair[1]

        # run the trials and return fitness in all trials
        trial_data = simulation_setup.run_joint_trials(agent1, agent2, simulation_setup.trials)

        # calculate overall fitness for a given trial run
        trial_fitness = harmonic_mean(trial_data['fitness'])
        # trial_fitness = np.mean(trial_data['fitness'])

        # update agent fitness with the current run
        # if it's the first run the fitness is just current trial fitness, otherwise it's a
        # cumulative average over all reshuffles up to now
        # if shuffle_num == 0:
        #     agent1.fitness = rolling_mean(0, trial_fitness, shuffle_num+1)
        #     agent2.fitness = rolling_mean(0, trial_fitness, shuffle_num+1)
        # else:
        #     agent1.fitness = rolling_mean(agent1.fitness, trial_fitness, shuffle_num+1)
        #     agent2.fitness = rolling_mean(agent2.fitness, trial_fitness, shuffle_num+1)
        if shuffle_num == 0:
            agent1.fitness = trial_fitness
            agent2.fitness = trial_fitness
        else:
            if trial_fitness > agent1.fitness:
                agent1.fitness = trial_fitness
            if trial_fitness > agent2.fitness:
                agent2.fitness = trial_fitness
        return agent1, agent2

    def create_population(self, size):
        """
        Create random population: used for creating a random initial population and random portion 
        of the new population in each generation.
        :param size: the size of the population to create
        :return: population of agents
        """
        population = []
        for i in range(size):
            # create the agent's CTRNN brain
            agent_brain = BrainCTRNN(self.network_params['num_neurons'],
                                     self.network_params['step_size'],
                                     self.network_params['tau_range'],
                                     self.network_params['g_range'],
                                     self.network_params['theta_range'],
                                     self.network_params['w_range'])

            if self.evaluation_params['velocity_control'] == "direct":
                agent = agents.DirectVelocityAgent(agent_brain, self.agent_params,
                                                   self.evaluation_params['screen_width'])
            else:
                # create new agent of a certain type
                # agent = agents.Agent(agent_brain, self.agent_params)
                # agent = agents.EmbodiedAgentV1(agent_brain, self.agent_params, self.evaluation_params['screen_width'])
                agent = agents.EmbodiedAgentV2(agent_brain, self.agent_params, self.evaluation_params['screen_width'])
                # agent = agents.ButtonOnOffAgent(agent_brain, self.agent_params,
                #         self.evaluation_params['screen_width'])
            population.append(agent)
        return population

    def create_joint_population(self, size):
        population = self.create_population(size)
        population_left = population[:int(size/2)]
        population_right = population[int(size/2):]
        return population_left, population_right

    def reproduce(self, population):
        """
        Reproduce a single generation in the following way:
        1) Copy the proportion equal to elitist_fraction of the current population to the new population
           (these are best_agents)
        2) Select part of the population for crossover using some selection method (set in config)
        3) Shuffle the selected population in preparation for cross-over
        4) Create crossover_fraction children of selected population with probability of crossover equal
           to prob_crossover.
        Crossover takes place at genome module boundaries (single neurons).
        5) Apply mutation to the children with mutation equal to mutation_var
        6) Fill the rest of the population with randomly created agents

        :param population: the population to be reproduced, already sorted in order of decreasing fitness
        :return: new_population
        """

        pop_size = len(population)
        new_population = [None] * pop_size

        # calculate all fractions
        n_best = math.floor(pop_size * self.evolution_params['elitist_fraction'] + 0.5)
        # floor to the nearest even number
        n_crossed = int(math.floor(pop_size * self.evolution_params['fps_fraction'] + 0.5)) & (-2)
        n_fillup = pop_size - (n_best + n_crossed)

        # 1) Elitist selection

        best_agents = deepcopy(population[:n_best])
        new_population[:n_best] = best_agents
        newpop_counter = n_best  # track where we are in the new population

        # 2) Select mating population from the remaining population

        updated_fitness = update_fitness(population, self.evolution_params['fitness_update'], 1.1)
        mating_pool = select_mating_pool(population, updated_fitness, n_crossed,
                                         self.evolution_params['selection'])

        # 3) Shuffle
        random.shuffle(mating_pool)

        # 4, 5) Create children with crossover or apply mutation
        mating_counter = 0
        mating_finish = newpop_counter + n_crossed

        while newpop_counter < mating_finish:
            if (mating_finish - newpop_counter) == 1:
                new_population[newpop_counter] = mutate(mating_pool[mating_counter],
                                                        self.evolution_params['mutation_variance'])
                newpop_counter += 1
                mating_counter += 1

            else:
                r = np.random.random()
                parent1 = mating_pool[mating_counter]
                parent2 = mating_pool[mating_counter + 1]

                # if r < self.evolution_params['prob_crossover'] and parent1 != parent2:
                if r < self.evolution_params['prob_crossover']:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    # if the two parents are the same, mutate them to get children
                    child1 = mutate(parent1, self.evolution_params['mutation_variance'])
                    child2 = mutate(parent2, self.evolution_params['mutation_variance'])
                new_population[newpop_counter], new_population[newpop_counter+1] = child1, child2
                newpop_counter += 2
                mating_counter += 2

        # 6) Fill up with random new agents
        new_population[newpop_counter:] = self.create_population(n_fillup)

        return new_population


class AgentPair:
    def __init__(self, agent1, agent2):
        self.left = agent1
        self.right = agent2
        self.fitness = 0


class JointEvolution:
    def __init__(self, pop_size, evolution_params, network_params, evaluation_params, agent_params):
        """
        Class that executes genetic search.
        :param pop_size: size of the new population
        :param evolution_params: evolution parameters for the simulation
        :param network_params: neural network parameters for the simulation
        :param evaluation_params: parameters used for running trials during the evaluation
        """
        self.pop_size = pop_size
        self.evolution_params = evolution_params
        self.step_size = network_params['step_size']
        self.network_params = network_params
        self.evaluation_params = evaluation_params
        self.agent_params = agent_params
        self.foldername = None

    def set_foldername(self, text):
        self.foldername = text

    def run_joint(self, gen_to_load, parallel_agents):
        """
        Execute a full search run until some condition is reached.
        :param gen_to_load: which generation to load if starting from an existing population
        :param parallel_agents: switch whether processing of agents within generation should be parallel
        :return: the last population in the search
        """
        gen = 0
        # create initial population or load existing one
        if gen_to_load is None:
            population_left, population_right = self.create_joint_population(self.pop_size)
            print("Created initial population...")
        else:
            # population_left, population_right = self.load_joint_population(self.foldername, gen_to_load)
            population_left, population_right = load_joint_population(gen_to_load)

        population = create_pairs(population_left, population_right, self.evolution_params['tested_proportion'])

        # collect average and best fitness
        avg_fitness = [0.0]
        best_fitness = [0.0]

        best_counter = 0

        # size of the left and right sub-populations
        num_pairs = int(len(population))

        # initialize a type of simulation
        simulation_setup = simulate.Simulation(self.step_size, self.evaluation_params)
        # simulation_run = simulate.SimpleSimulation(evolution.step_size, evolution.evaluation_params)

        while gen < self.evolution_params['max_gens'] + 1:
            print(gen)
            tested_pairs = []
            # evaluate all agents on the task
            if parallel_agents:
                num_cores = self.evolution_params['num_cores']
                slice_size = int(num_pairs/num_cores)
                for slice_counter in range(slice_size):
                    pairs_slice = population[slice_counter*num_cores:(slice_counter+1)*num_cores]
                    # updated_slice = Parallel(n_jobs=num_cores, prefer="threads")(delayed(self.process_pair)
                    updated_slice = Parallel(n_jobs=num_cores)(delayed(self.process_pair)
                                                               (pair, simulation_setup)
                                                               for pair in pairs_slice)
                    tested_pairs.extend(updated_slice)
            else:
                for pair in population:
                    tested_pairs.extend(self.process_pair(pair, simulation_setup))

            # log fitness results: average population fitness
            avgf = np.mean(pop_fitness(tested_pairs)).item()
            avg_fitness.append(round(avgf, 3))

            # sort agents by fitness from best to worst
            tested_pairs.sort(key=lambda pair: pair.fitness, reverse=True)

            # log fitness results: best agent fitness
            bf = round(tested_pairs[0].fitness, 3)
            best_fitness.append(bf)

            # stop the search if fitness hasn't increased in a set number of generations
            if bf == best_fitness[-2]:
                best_counter += 1

                if best_counter > self.evolution_params['evolution_break']:
                    # save the last population
                    save_population(tested_pairs, self.foldername, gen)
                    # print("Stopped the search at generation {}".format(gen))

                    # save the average and best fitness lists
                    fitness_log = {'average': avg_fitness, 'best': best_fitness}
                    log_fitness(self.foldername, fitness_log)
                    break
            else:
                best_counter = 0

            # save the intermediate or last population and fitness
            if gen % self.evolution_params['check_int'] == 0 or gen == self.evolution_params['max_gens']:
                save_population(tested_pairs, self.foldername, gen)
                # print("Saved generation {}".format(gen))
                fitness_log = {'average': avg_fitness, 'best': best_fitness}
                log_fitness(self.foldername, fitness_log)

            # reproduce population
            population = self.reproduce(tested_pairs)
            gen += 1

    @staticmethod
    def process_pair(pair, simulation_setup):
        agent1 = pair.left
        agent2 = pair.right

        # run the trials and return fitness in all trials
        trial_data = simulation_setup.run_joint_trials(agent1, agent2, simulation_setup.trials)

        # calculate overall fitness for a given trial run
        trial_fitness = harmonic_mean(trial_data['fitness'])
        # trial_fitness = np.mean(trial_data['fitness'])
        pair.fitness = trial_fitness
        print(agent1.name, agent2.name, trial_fitness)
        return pair

    def create_population(self, size):
        """
        Create random population: used for creating a random initial population and random portion
        of the new population in each generation.
        :param size: the size of the population to create
        :return: population of agents
        """
        population = []
        for i in range(size):
            # create the agent's CTRNN brain
            agent_brain = BrainCTRNN(self.network_params['num_neurons'],
                                     self.network_params['step_size'],
                                     self.network_params['tau_range'],
                                     self.network_params['g_range'],
                                     self.network_params['theta_range'],
                                     self.network_params['w_range'])

            if self.evaluation_params['velocity_control'] == "direct":
                agent = agents.DirectVelocityAgent(agent_brain, self.agent_params,
                                                   self.evaluation_params['screen_width'])
            else:
                # create new agent of a certain type
                # agent = agents.Agent(agent_brain, self.agent_params)
                # agent = agents.EmbodiedAgentV1(agent_brain, self.agent_params, self.evaluation_params['screen_width'])
                agent = agents.EmbodiedAgentV2(agent_brain, self.agent_params, self.evaluation_params['screen_width'])
                # agent = agents.ButtonOnOffAgent(agent_brain, self.agent_params,
                # self.evaluation_params['screen_width'])
            population.append(agent)
        return population

    def create_joint_population(self, size):
        population = self.create_population(size)
        population_left = population[:int(size/2)]
        population_right = population[int(size/2):]
        return population_left, population_right

    def reproduce(self, population):
        """
        Reproduce a single generation in the following way:
        1) Copy the proportion equal to elitist_fraction of the current population to the new population
           (these are best_agents)
        2) Select part of the population for crossover using some selection method (set in config)
        3) Shuffle the selected population in preparation for cross-over
        4) Create crossover_fraction children of selected population with probability of crossover equal
           to prob_crossover.
        Crossover takes place at genome module boundaries (single neurons).
        5) Apply mutation to the children with mutation equal to mutation_var
        6) Fill the rest of the population with randomly created agents

        :param population: the population to be reproduced, already sorted in order of decreasing fitness
        :return: new_population
        """

        pop_size = len(population)  # this will be 150
        new_population = [None] * pop_size

        # calculate all fractions
        n_best = math.floor(pop_size * self.evolution_params['elitist_fraction'] + 0.5)
        # floor to the nearest even number
        n_crossed = int(math.floor(pop_size * self.evolution_params['fps_fraction'] + 0.5)) & (-2)
        n_fillup = pop_size - (n_best + n_crossed)

        # 1) Elitist selection
        # Preserve the best pairs as they are
        best_pairs = deepcopy(population[:n_best])
        new_population[:n_best] = best_pairs
        newpop_counter = n_best  # track where we are in the new population

        # 2) Select mating population from the remaining population
        updated_fitness = update_fitness(population, self.evolution_params['fitness_update'], 1.1)
        mating_pool = select_mating_pool(population, updated_fitness, n_crossed, self.evolution_params['selection'])

        # 3) Shuffle
        random.shuffle(mating_pool)

        mating_left = [pair.left for pair in mating_pool]
        mating_right = [pair.right for pair in mating_pool]

        # 4, 5) Create children with crossover or apply mutation
        mating_counter = 0
        mating_finish = newpop_counter + n_crossed

        while newpop_counter < mating_finish:
            if (mating_finish - newpop_counter) == 1:
                new_left = mutate(mating_left[mating_counter], self.evolution_params['mutation_variance'])
                new_right = mutate(mating_right[mating_counter], self.evolution_params['mutation_variance'])

                new_population[newpop_counter] = AgentPair(new_left, new_right)
                newpop_counter += 1
                mating_counter += 1

            else:
                r = np.random.random()
                parent1_left = mating_left[mating_counter]
                parent2_left = mating_left[mating_counter + 1]

                parent1_right = mating_right[mating_counter]
                parent2_right = mating_right[mating_counter + 1]

                # if r < self.evolution_params['prob_crossover'] and parent1 != parent2:
                if r < self.evolution_params['prob_crossover']:
                    child1_left, child2_left = crossover(parent1_left, parent2_left)
                    child1_right, child2_right = crossover(parent1_right, parent2_right)
                else:
                    child1_left = mutate(parent1_left, self.evolution_params['mutation_variance'])
                    child2_left = mutate(parent2_left, self.evolution_params['mutation_variance'])
                    child1_right = mutate(parent1_right, self.evolution_params['mutation_variance'])
                    child2_right = mutate(parent2_right, self.evolution_params['mutation_variance'])

                new_population[newpop_counter], new_population[newpop_counter+1] = AgentPair(child1_left,
                                                                                             child1_right),\
                    AgentPair(child2_left, child2_right)
                newpop_counter += 2
                mating_counter += 2

        # 6) Fill up with random pairs
        # TODO: new combinations of existing agents or new agents?
        random.shuffle(mating_right)
        random_pairs = list(zip(mating_left, mating_right))
        new_pairs = []
        for pair in random_pairs:
            new_pairs.append(AgentPair(pair[0], pair[1]))
        new_population[newpop_counter:] = new_pairs[:n_fillup]

        return new_population


def load_population(foldername, gen):
    pop_file = open('{}/gen{}'.format(foldername, gen), 'rb')
    population = pickle.load(pop_file)
    pop_file.close()
    population.sort(key=lambda agent: agent.fitness, reverse=True)
    return population


def load_joint_population(genfile):
    pop_file = open(genfile, 'rb')
    population = pickle.load(pop_file)
    pop_file.close()
    random.shuffle(population)
    size = len(population)
    # population_left = population[:int(size/2)]
    # population_right = population[int(size/2):]
    population_left = deepcopy(population)
    population_right = deepcopy(population)
    return population_left, population_right


def save_population(population, foldername, gen):
    pop_file = open('{}/gen{}'.format(foldername, gen), 'wb')
    pickle.dump(population, pop_file)
    pop_file.close()


def log_fitness(foldername, fits):
    fit_file = open('{}/fitnesses'.format(foldername), 'wb')
    pickle.dump(fits, fit_file)
    fit_file.close()


def make_rand_vector(dims):
    """
    Generate a random unit vector.  This works by first generating a vector each of whose elements
    is a random Gaussian and then normalizing the resulting vector.
    """
    vec = np.random.normal(0, 1, dims)
    mag = sum(vec ** 2) ** .5
    return vec/mag


# TODO: consider replacing 0 values with small positive number to avoid slashing everything to 0
def harmonic_mean(fit_list):
    if 0 in fit_list:
        return 0
    else:
        return len(fit_list) / np.sum(1.0 / np.array(fit_list))


def rolling_mean(prev, current, counter):
    return (prev * (counter - 1) + current) / counter


def pop_fitness(population):
    return [agent.fitness for agent in population]


def update_fitness(population, method, max_exp_offspring=None):
    """
    Update agent fitness to relative values, retain sorting from best to worst.
    :param population: the population whose fitness needs updating
    :param method: fitness proportionate or rank-based
    :param max_exp_offspring:
    :return:
    """
    rel_fitness = []
    if method == 'fps':
        fitnesses = [agent.fitness for agent in population]
        total_fitness = float(sum(fitnesses))
        rel_fitness = [f / total_fitness for f in fitnesses]

    elif method == 'rank':
        # Baker's linear ranking method: f(pos) = 2-SP+2*(SP-1)*(pos-1)/(n-1)
        # the highest ranked individual receives max_exp_offspring (typically 1.1),
        # the lowest receives 2 - max_exp_offspring
        # normalized to sum to 1
        ranks = list(range(1, len(population) + 1))
        rel_fitness = [(max_exp_offspring + (2 - 2 * max_exp_offspring) * (ranks[i] - 1) /
                        (len(population) - 1)) / len(population)
                       for i in range(len(population))]

    elif method == 'sigma':
        # for every individual 1 + (I(f) - P(avg_f))/2*P(std) is calculated
        # if value is below zero, a small positive constant is given so the individual has some probability
        # of being chosen. The numbers are then normalized
        fitnesses = [agent.fitness for agent in population]
        avg = np.mean(fitnesses).item()
        std = max(0.0001, np.std(fitnesses).item())
        exp_values = list((1 + ((f - avg) / (2 * std))) for f in fitnesses)

        for i, v in enumerate(exp_values):
            if v <= 0:
                exp_values[i] = 1 / len(population)
        s = sum(exp_values)
        rel_fitness = list(e / s for e in exp_values)

    return rel_fitness


def select_mating_pool(population, updated_fitness, n_parents, method):
    """
    Select a mating pool population.
    :param population: the population from which to select the parents
    :param updated_fitness: the relative updated fitness
    :param n_parents: how many parents to select
    :param method: which method to use for selection (roulette wheel or stochastic universal sampling)
    :return: selected parents for reproduction
    """
    new_population = []

    if method == "rws":
        # roulette wheel selection
        probs = [sum(updated_fitness[:i + 1]) for i in range(len(updated_fitness))]
        # Draw new population
        new_population = []
        for _ in range(n_parents):
            r = np.random.random()
            for (i, agent) in enumerate(population):
                if r <= probs[i]:
                    new_population.append(agent)
                    break

    elif method == "sus":
        # stochastic universal sampling selection
        probs = [sum(updated_fitness[:i + 1]) for i in range(len(updated_fitness))]
        p_dist = 1 / n_parents  # distance between the pointers
        start = np.random.uniform(0, p_dist)
        pointers = [start + i * p_dist for i in range(n_parents)]

        for p in pointers:
            for (i, agent) in enumerate(population):
                if p <= probs[i]:
                    new_population.append(agent)
                    break

    return new_population


def crossover(parent1, parent2):
    """
    Given two agents, create two new agents by exchanging their genetic material.
    :param parent1: first parent agent
    :param parent2: second parent agent
    :return: two new agents
    """
    crossover_point = np.random.choice(parent1.crossover_points)
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    child1.genotype = np.hstack((parent1.genotype[:crossover_point], parent2.genotype[crossover_point:]))
    child2.genotype = np.hstack((parent2.genotype[:crossover_point], parent1.genotype[crossover_point:]))
    child1.make_params_from_genotype(child1.genotype)
    child2.make_params_from_genotype(child2.genotype)
    return child1, child2


def mutate(agent, mutation_var):
    magnitude = np.random.normal(0, np.sqrt(mutation_var))
    unit_vector = make_rand_vector(len(agent.genotype))
    mutant = deepcopy(agent)
    mutant.genotype = np.clip(agent.genotype + magnitude * unit_vector,
                              agent.gene_range[0], agent.gene_range[1])
    mutant.make_params_from_genotype(mutant.genotype)
    return mutant


def create_pairs(population_left, population_right, num_copies):
    # this will create pop_size / 2 * tested_shuffles number of pairs
    all_pairs = []
    shuffle_counter = 0
    while shuffle_counter < num_copies:
        random.shuffle(population_right)
        pairs_to_test = list(zip(population_left, population_right))
        all_pairs.extend(pairs_to_test)
        shuffle_counter += 1
    pair_objects = []
    for pair in all_pairs:
        pair_objects.append(AgentPair(pair[0], pair[1]))
    return pair_objects
