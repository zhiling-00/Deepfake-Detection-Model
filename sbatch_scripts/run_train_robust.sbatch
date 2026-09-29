#!/bin/bash
#SBATCH --job-name=train_robust
#SBATCH -G 1
#SBATCH --export=ALL,PATH=/media/D1/ait_users/staff122/.conda/envs/deepfake/bin:/usr/local/bin:/usr/bin:/bin
#SBATCH --output=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/train_robust_%j.out
#SBATCH --error=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/train_robust_%j.err

cd /media/D1/ait_users/staff122/deepfake_project

python deepfake_detection_efficientnet.py train_robust --data_root /media/D1/ait_users/staff122/deepfake_project/FaceForensics++_C23 --output_root /media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project --data_format ffpp --model efficientnet_b4 --img_size 380 --batch_size 8 --epochs 10 --num_workers 2
