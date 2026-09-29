#!/bin/bash
#SBATCH --job-name=cross_celeb_cons
#SBATCH -G 1
#SBATCH --export=ALL,PATH=/media/D1/ait_users/staff122/.conda/envs/deepfake/bin:/usr/local/bin:/usr/bin:/bin
#SBATCH --output=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/cross_celeb_consistency_%j.out
#SBATCH --error=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/cross_celeb_consistency_%j.err

cd /media/D1/ait_users/staff122/deepfake_project

python cross_evaluation.py --data_root /media/D1/ait_users/staff122/deepfake_project/FaceForensics++_C23 --output_root /media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project --external_root /media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/Celeb_V2 --dataset_name celeb_v2 --model efficientnet_b4 --img_size 380 --batch_size 8 --num_workers 2
