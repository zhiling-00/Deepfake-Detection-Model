#!/bin/bash
#SBATCH --job-name=dfdc_subset77
#SBATCH -G 1
#SBATCH --export=ALL,PATH=/media/D1/ait_users/staff122/.conda/envs/deepfake/bin:/usr/local/bin:/usr/bin:/bin
#SBATCH --output=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/dfdc_subset77_%j.out
#SBATCH --error=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/dfdc_subset77_%j.err

cd /media/D1/ait_users/staff122/deepfake_project

python dfdc_video_preprocess.py
