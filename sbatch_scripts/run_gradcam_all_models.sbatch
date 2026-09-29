#!/bin/bash
#SBATCH --job-name=gradcam_all
#SBATCH -G 1
#SBATCH --export=ALL,PATH=/media/D1/ait_users/staff122/.conda/envs/deepfake/bin:/usr/local/bin:/usr/bin:/bin
#SBATCH --output=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/gradcam_all_%j.out
#SBATCH --error=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/gradcam_all_%j.err

cd /media/D1/ait_users/staff122/deepfake_project

python generate_gradcam.py --data_root /media/D1/ait_users/staff122/deepfake_project/FaceForensics++_C23 --output_root /media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project --model efficientnet_b4 --img_size 380 --num_per_class 5
