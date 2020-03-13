from analyzer import analyze as az
import os
import fnmatch
import argparse
from dotenv import load_dotenv


load_dotenv()


def main(condition, agent_type, pop_type, seed_num):
    agent_directory = os.path.join(os.getenv("DATA_DIR"), condition, agent_type, seed_num)
    gen_files = fnmatch.filter(os.listdir(agent_directory), 'gen*')
    gen_numbers = [int(x[3:]) for x in gen_files]
    last_gen = max(gen_numbers)

    if condition == "single":
        if pop_type == "random":
            # random single agents
            td, ag = az.run_random_population(1, "all")
        else:
            # check evolved agents
            az.plot_fitness(agent_directory)
            td, ag = az.run_single_agent('single', agent_type, seed_num, last_gen, 0, "all")
            # az.check_generalization('single', agent_type, seed_num, ag)

            # # additional checks
            # w = az.plot_weights('single', 'buttons', 123, [0, 10, 20, 30, 40], 1)
            # az.animate_trial('single', 'buttons', 123, 0, 0, 0)

    elif condition == "joint":
        if pop_type == "random":
            # random joint agents
            td, a1, a2 = az.run_random_pair(5, 'all')
        else:
            # check evolved joint agents
            # agent_type, seed, generation_num, agent_num, to_plot
            az.plot_fitness(agent_directory)
            # td1, a1, a2 = az.run_single_pair(agent_type, seed_num, last_gen, 0, 'all')
            td1, a1, a2 = az.plot_best_pair(agent_directory, last_gen, 'all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", type=str, help="specify the condition",
                        choices=["single", "joint"])
    parser.add_argument("agent_type", type=str, help="specify the type of the agent you want to run",
                        choices=["buttons", "direct"])
    parser.add_argument("pop_type", type=str, help="specify the type of population you want to run",
                        choices=["seeded", "random"])
    parser.add_argument("seed_num", type=str, help="specify random seed number")
    args = parser.parse_args()
    main(args.condition, args.agent_type, args.pop_type, args.seed_num)
