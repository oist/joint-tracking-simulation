#!/bin/bash
module load python/3.7.3
cd $HOME/joint-tracking-simulation
source ./env/bin/activate
python main_joint.py joint direct 111
