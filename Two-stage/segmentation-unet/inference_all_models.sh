#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /hhome/priubrogent/tfg/segmenter-unet/logs # working directory
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 40048 # 12GB solicitados.
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 

# python /hhome/priubrogent/tfg/segmenter-unet/code/test_segmentation_inference.py --model /test_models/big_dataset_r2attunetreduced_60_v1.pth --name "r2attunetreduced"
# python /hhome/priubrogent/tfg/segmenter-unet/code/test_segmentation_inference.py --model /test_models/big_dataset_unet_50_v1.pth --name "unet"
# python /hhome/priubrogent/tfg/segmenter-unet/code/test_segmentation_inference.py --model /test_models/pretrained_unet_40_epoch_oldloader_v1.pth --name "unet"
# python /hhome/priubrogent/tfg/segmenter-unet/code/test_segmentation_inference.py --model /test_models/big_dataset_attunet_40_v1.pth --name "attunet"
# python /hhome/priubrogent/tfg/segmenter-unet/code/test_segmentation_inference.py --model /test_models/big_dataset_unet_new_loader_unet_40.pth --name "unet" --nou_loader
python /hhome/priubrogent/tfg/segmenter-unet/code/test_segmentation_inference_bo.py --model /final_models/final_pretrained_denoise_unet.pth --name "unet" --input /input
# python /hhome/priubrogent/tfg/segmenter-unet/code/test_segmentation_inference_bo.py --model /final_models/final_pretrained_docum_attunet_BO_attunet_40epochs_1338images_final.pth --name "attunet" --input /input
# python /hhome/priubrogent/tfg/segmenter-unet/code/test_segmentation_inference_bo.py --model /final_models/final_pretrained_docum_r2attunet_BO_r2attunet_60epochs_1338images_final.pth --name "r2attunet" --input /input
python /hhome/priubrogent/tfg/segmenter-unet/code/test_segmentation_inference_bo.py --model /final_models/final_pretrained_denoise_attunet.pth --name "attunet" --input /input






