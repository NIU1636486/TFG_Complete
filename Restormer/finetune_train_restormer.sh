#!/usr/bin/env bash
#!/bin/bash
#SBATCH -n 8 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /hhome/priubrogent/tfg/Restormer/ # working directory
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 40048 # 12GB solicitados.
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1

python /hhome/priubrogent/tfg/Restormer/basicsr/train.py -opt /hhome/priubrogent/tfg/Restormer/Film_Restormer_finetunning.yml --wandb