import os
import fnmatch
from ..analyzer import analyze as az

agent_directory = "agents/single/direct/102575"
gen_files = fnmatch.filter(os.listdir(agent_directory), 'gen*')
gen_numbers = [int(x[3:]) for x in gen_files]
last_gen = max(gen_numbers)

# Get trial and agent data
config = az.load_config("single", "direct", "102575")
population = az.load_population("single", "direct", "102575", last_gen)
ag = population[0]
print(ag.brain.W)
