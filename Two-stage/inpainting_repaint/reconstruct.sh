#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /hhome/priubrogent/tfg/inpainting/RePaint # working directory
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 40048 # 12GB solicitados.
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 

python /hhome/priubrogent/tfg/inpainting/RePaint/reconstruct.py --blocks-dir ./resultats --output ./reconstructed.png --overlap 32 --block-info ./resultats/bueno_feo_malo_frame_0129_block_info.txt --block-type inpainted
python /hhome/priubrogent/tfg/inpainting/RePaint/reconstruct.py --blocks-dir ./resultats --no-blend --output ./reconstructed_noblend.png --overlap 32 --block-info ./resultats/bueno_feo_malo_frame_0129_block_info.txt --block-type inpainted
