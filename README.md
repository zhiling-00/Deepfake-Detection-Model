# Explainable and Robust Deepfake Detection using EfficientNet-B4
Proposed a Degradation-Consistency  EfficientNet-B4 model which provided visual explanations using Grad-CAM and considered robustness when evaluating the model.

This project implements an explainable and robustness-focused deepfake detection pipeline using EfficientNet-B4. The model is trained on FaceForensics++ C23 and evaluated under both within-domain degradation testing and clean-only cross-domain testing.

The project compares three EfficientNet-B4-based models:

1. **Baseline EfficientNet-B4**
   A clean-trained EfficientNet-B4 binary classifier for real/fake face frame classification.
2. **DA-EfficientNet-B4** **(Degradation-Aware EfficientNet-B4)**
   An EfficientNet-B4 model trained with random degradation-aware augmentation, including clean, JPEG compression, blur, downscale, and noise.
3. **DC-EfficientNet-B4 (Degradation-Consistency EfficientNet-B4)**
   The final proposed consistency-based model. In the code, this is implemented using `train_consistency` and saved as `consistency_efficientnet_b4_robust_best.pt`. It fine-tunes from the baseline checkpoint using clean-degraded image pairs, a deeper classifier head, and a consistency loss.

The project also includes Grad-CAM visualization to support explainability by showing which facial regions the models focus on when making predictions.

---

## 1. Project Structure

```text
deepfake_project/
├── deepfake_detection_efficientnet.py
├── cross_evaluation.py
├── evaluate_deepfakebench.py
├── generate_gradcam.py
├── dfdc_video_preprocess.py
├── README.md
└── outputs_deepfake_project/
    ├── models/
    ├── results/
    ├── gradcam/
└── datasets/
    ├── FaceForensics++_C23/
    ├── Celeb-DF-v2/
    ├── DFDC/
└── sbatch_scripts/
└── logs/
```

---

## 2. Python File Descriptions

### 2.1 `deepfake_detection_efficientnet.py` — Main Pipeline

This is the **main file** of the project.

It handles:

- FaceForensics++ C23 video collection
- Video-level train/validation/test split
- Frame extraction
- Haar Cascade face cropping
- Clean JPEG preprocessing
- Baseline EfficientNet-B4 training
- DA-EfficientNet-B4 training
- DC-EfficientNet-B4 training
- Within-domain evaluation under clean and degraded conditions

Supported modes:

```text
prepare
check
train_baseline
train_robust
train_consistency
eval_baseline
eval_robust
eval_consistency
full
```

Model meanings:

| Code Mode             | Model                                                   |
| --------------------- | ------------------------------------------------------- |
| `train_baseline`    | Baseline EfficientNet-B4                                |
| `train_robust`      | DA-EfficientNet-B4                                      |
| `train_consistency` | DC-EfficientNet-B4                                      |
| `eval_baseline`     | Evaluate baseline under clean/JPEG/blur/downscale/noise |
| `eval_robust`       | Evaluate DA model under clean/JPEG/blur/downscale/noise |
| `eval_consistency`  | Evaluate DC model under clean/JPEG/blur/downscale/noise |

---

### 2.2 `cross_evaluation.py` — Clean-Only Cross-Domain Evaluation

This file evaluates trained project models on external datasets such as Celeb-DF-v2 and the DFDC subset.

It expects external datasets to follow this structure:

```text
DatasetRoot/
└── test/
    ├── real/
    └── fake/
```

or:

```text
DatasetRoot/
└── Test/
    ├── real/
    └── fake/
```

It can convert supported image formats such as PNG, JPEG, BMP, WEBP, TIF, and TIFF into JPG-compatible folders for the main pipeline loader.

This script evaluates:

- Baseline EfficientNet-B4
- DA-EfficientNet-B4 if `--include_old_da` is used
- DC-EfficientNet-B4

Cross-domain testing is clean-only because the purpose is to test generalization to unseen datasets, not robustness to artificial degradation.

---

### 2.3 `evaluate_deepfakebench.py` — DeepfakeBench External Baseline Evaluation

This file evaluates selected DeepfakeBench models on the same prepared datasets.

Supported DeepfakeBench detectors:

- EfficientNet-B4
- XceptionNet

This script keeps the original DeepfakeBench detector structure unchanged:

```text
image → backbone.features() → backbone.classifier() → two-class logits
fake_probability = softmax(logits)[:, 1]
```

It only adapts the evaluation side so that DeepfakeBench models can be tested on this project's prepared `test/real` and `test/fake` image folders.

This file is used for fair external baseline comparison. The user must provide the correct trained DeepfakeBench checkpoint weights, such as:

```text
effnb4_best.pth
xception_best.pth
```

---

### 2.4 `generate_gradcam.py` — Grad-CAM Explainability

This file generates Grad-CAM visual explanations for trained models.

It supports:

- Baseline EfficientNet-B4
- DA-EfficientNet-B4
- DC-EfficientNet-B4

The script selects real and fake test frames from the preprocessed FaceForensics++ C23 test set and generates:

- Original image
- Grad-CAM heatmap
- Overlay image

The target layer is the last EfficientNet feature block:

```text
model.features[-1]
```

The output images are saved under:

```text
outputs_deepfake_project/gradcam/
```

Grad-CAM is used to check whether the model focuses on meaningful facial regions instead of irrelevant background regions.

---

### 2.5 `dfdc_video_preprocess.py` — DFDC Subset Preparation

This file creates a balanced DFDC subset for cross-domain evaluation.

Default setup:

```text
n_real = 77
n_fake = 77
frames_per_video = 32
img_size = 380
```

The output folder structure is:

```text
DFDC_subset_77_frames/
└── test/
    ├── real/
    └── fake/
```

This processed DFDC folder can then be used with `cross_evaluation.py`.

---

## 3. Installation Environment

### 3.1 Recommended Python Version

```text
Python 3.10
```

### 3.2 Create Conda Environment

```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake
```

### 3.3 Install Required Packages

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio
```

Then install the remaining packages:

```bash
pip install opencv-python pillow numpy pandas scikit-learn tqdm matplotlib pyyaml
```

For DeepfakeBench EfficientNet-B4 evaluation, install:

```bash
pip install efficientnet-pytorch
```

### 3.4 Main Dependencies

| Package      | Purpose                                                      |
| ------------ | ------------------------------------------------------------ |
| PyTorch      | Model training and inference                                 |
| Torchvision  | EfficientNet-B4 backbone                                     |
| OpenCV       | Video reading, face detection, image processing              |
| Pillow       | Image loading and conversion                                 |
| NumPy        | Numerical processing                                         |
| Pandas       | CSV metadata and result storage                              |
| Scikit-learn | Accuracy, precision, recall, F1-score, AUC, confusion matrix |
| Matplotlib   | Grad-CAM visualization                                       |
| PyYAML       | DeepfakeBench config loading                                 |
| tqdm         | Progress bars                                                |

---

## 4. Recommended Running Order

```text
1. Prepare FaceForensics++ C23 frames
2. Check preprocessed folders
3. Train Baseline EfficientNet-B4
4. Train DA-EfficientNet-B4
5. Train DC-EfficientNet-B4
6. Evaluate all three project models within-domain
7. Prepare external datasets
8. Run clean-only cross-domain evaluation
9. Run DeepfakeBench external baseline evaluation
10. Generate Grad-CAM visualizations
```

---

## 5. Output Files

Important output files:

| File                                                | Description                          |
| --------------------------------------------------- | ------------------------------------ |
| `video_split_FaceForensics++_C23.csv`             | Video-level train/val/test split     |
| `frames_metadata_FaceForensics++_C23.csv`         | Extracted frame metadata             |
| `baseline_efficientnet_b4_clean_best.pt`          | Best baseline model                  |
| `da_efficientnet_b4_robust_best.pt`               | Best DA model                        |
| `consistency_efficientnet_b4_robust_best.pt`      | Best DC model                        |
| `*_degradation_results_medium.csv`                | Within-domain degradation evaluation |
| `*_clean_cross_domain_comparison_consistency.csv` | Cross-domain clean-only evaluation   |
| `*_summary_medium.csv`                            | DeepfakeBench evaluation summary     |

---

## 6. Evaluation Metrics

The main reported metric is:

```text
AUC
```

AUC is used because it measures how well the model separates real and fake samples based on predicted fake probabilities without depending only on a fixed 0.5 threshold.

Supporting metrics include:

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
Loss
```

F1-score is useful because it balances precision and recall. Accuracy is included but should not be used alone, especially when the number of real and fake frames is imbalanced.
