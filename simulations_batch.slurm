#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem=8G
#SBATCH -c 4

module load python/3.7.3
cd $HOME/oist_code/joint-tracking-simulation
source ./env/bin/activate
python main_joint.py direct 914463