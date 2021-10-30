#!/bin/zsh
#SBATCH -p long
#SBATCH --mem=32g
#SBATCH -c 2
#SBATCH --output=/home/broerman/projects/crispy_shifty/projects/crispy_shifty_dimers/02_design_filter/head_node.out

source activate /projects/crispy_shifty/envs/crispy

python /home/broerman/projects/crispy_shifty/projects/crispy_shifty_dimers/02_design_filter.py
