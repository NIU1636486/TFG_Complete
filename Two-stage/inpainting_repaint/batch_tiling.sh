#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /hhome/priubrogent/tfg/inpainting/RePaint # working directory
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 40048 # 12GB solicitados.
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 

python batch_tiled.py \
    --input-folder ./repaint/input/ \
    --mask-folder ./repaint/results_rescaled/final_attunet/ \
    --output-folder ./reconstructed_results_final_attunet/ \
    --conf_path ./confs/film_blocks.yml \
    --gt-folder ./repaint/target/\
    --overlap 32