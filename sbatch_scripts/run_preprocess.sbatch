#!/bin/bash
#SBATCH --job-name=deepfake_preprocess
#SBATCH -G 1
#SBATCH --export=ALL,PATH=/media/D1/ait_users/staff122/.conda/envs/deepfake/bin:/usr/local/bin:/usr/bin:/bin
#SBATCH --output=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/preprocess_%j.out
#SBATCH --error=/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/preprocess_%j.err

cd /media/D1/ait_users/staff122/deepfake_project

python deepfake_detection_efficientnet.py prepare --data_root /media/D1/ait_users/staff122/deepfake_project/FaceForensics++_C23 --output_root /media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project --data_format ffpp --frames_per_video 32 --face_detector haar
