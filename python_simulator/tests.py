import numpy as np
from python_simulator import simulate
from python_simulator import evolve
import pickle
import time

# evolution parameters
pop_size = 10
max_gens = 10  # maximum generations to evolve
mutation_var = 1
prob_crossover = 0.8
elitist_frac = 0.1
fps_frac = 0.8
check_int = 5  # interval (in generations) of how often to dump the current search state

# network parameters
n_neurons = 8
step_size = 0.01
tau_range = (1, 10)
theta_range = (-15, 15)
w_range = (-15, 15)
g_range = (1, 1)

# evaluation parameters
velocities = [3.3, 4.3, -3.3, -4.3]
impact = [0.7, 1.0]
screen_width = [-20, 20]

evolution_params = [max_gens, mutation_var, prob_crossover, elitist_frac, fps_frac, check_int]
network_params = [n_neurons, step_size, tau_range, theta_range, w_range, g_range]
evaluation_params = [screen_width, velocities, impact]

evolution = evolve.Evolution(pop_size, evolution_params, network_params, evaluation_params)
population = evolution.create_population(pop_size)


def test_single_trial(tested_pop):
    agent = tested_pop[0]
    # calculate and make sure all works fine
    tau = agent.brain.Tau
    theta = agent.brain.Theta
    g = agent.brain.G
    w = agent.brain.W

    simulation_run = simulate.Simulation(step_size, evaluation_params)
    # returns a list of fitness in all trials
    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)

    tgp1 = trial_data['target_pos'][0]
    trp1 = trial_data['tracker_pos'][0]
    trv1 = trial_data['tracker_v'][0]
    io1 = trial_data['input-output'][0]
    br1 = trial_data['brain_state'][0]
    kp1 = trial_data['keypress'][0]

    tgp2 = trial_data['target_pos'][1]
    trp2 = trial_data['tracker_pos'][1]
    trv2 = trial_data['tracker_v'][1]
    io2 = trial_data['input-output'][1]
    br2 = trial_data['brain_state'][1]
    kp2 = trial_data['keypress'][1]

    o = sigmoid(np.multiply(g, br1 + theta))
    np.hstack((agent.VW, agent.AW, agent.MW))
    input = np.zeros(n_neurons)
    input[7] = io1[0] * trp1
    input[1] = io1[1] * tgp1
    input[0] = np.sum([io1[2] * tgp1, io1[3] * trp1])

    dy_dt = np.multiply(1 / tau, - br1 + np.dot(w, o) + input) * step_size
    y = br1 + dy_dt

    n4out = br1[3]
    n6out = br1[5]

    activation_left = np.sum([n4out * io1[8], n6out * io1[10]])
    activation_right = np.sum([n4out * io1[9], n6out * io1[11]])

    # measure time taken
    start_time = time.time()
    simulation_run.run_trials(agent, simulation_run.trials)
    elapsed_time = time.time() - start_time
    print(elapsed_time)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_population_trial(tested_pop):
    fitnesses = []
    for agent in tested_pop:
        simulation_run = simulate.Simulation(step_size, evaluation_params)
        # returns a list of fitness in all trials
        trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)
        agent.fitness = np.mean(trial_data['fitness'])
        fitnesses.append(agent.fitness)
    return fitnesses


def test_n_populations(tested_pop):
    avg_fit = []
    best_fit = []

    gen = 0
    while gen < max_gens:
        print('Generation {}'.format(gen))
        for agent in tested_pop:
            simulation_run = simulate.Simulation(step_size, evaluation_params)
            trial_data = simulation_run.run_trials(agent, simulation_run.trials)
            agent.fitness = np.mean(trial_data['fitness'])

        # log fitness results
        population_avg_fitness = np.mean(evolve.pop_fitness(tested_pop)).item()
        avg_fit.append(round(population_avg_fitness, 3))
        print("Average fitness in generation {} is {}".format(gen, round(population_avg_fitness, 3)))

        # sort agents by fitness from best to worst
        tested_pop.sort(key=lambda ag: ag.fitness, reverse=True)
        # log fitness results: best agent fitness
        bf = round(tested_pop[0].fitness, 3)
        best_fit.append(bf)

        new_population = evolution.reproduce(tested_pop)

        # save the intermediate population
        if gen % check_int == 0:
            popfile = open('./agents/gen{}'.format(gen), 'wb')
            pickle.dump(tested_pop, popfile)
            popfile.close()

        tested_pop = new_population
        gen += 1

    fits = [avg_fit, best_fit]
    fitfile = open('./agents/fitnesses', 'wb')
    pickle.dump(fits, fitfile)
    fitfile.close()
