#!/bin/bash

sbatch normal_train_restormer.sh
sbatch finetune_train_restormer.sh
sbatch bigpatch_train_restormer.sh
