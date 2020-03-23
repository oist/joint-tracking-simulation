#!/bin/bash
#SBATCH -t 2-20
#SBATCH --mem=8G
#SBATCH -c 20

module load python/3.7.3
cd $HOME/joint-tracking-simulation
source ./env/bin/activate
python main_joint.py joint direct 914463
