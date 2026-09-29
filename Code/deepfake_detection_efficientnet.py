# Dataset (FaceForensics++ C23): https://www.kaggle.com/datasets/xdxd003/ff-c23

import os
import cv2
import json
import random
import warnings
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# 1. Configuration
@dataclass
class CFG:
    DATA_ROOT: str = "/media/D1/ait_users/staff122/deepfake_project/FaceForensics++_C23"
    OUTPUT_ROOT: str = "/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project_fixed"
    DATA_FORMAT: str = "ffpp"

    REAL_DIR_NAME: str = "real"
    FAKE_DIR_NAME: str = "fake"

    # FaceForensics++ format
    FFPP_REAL_DIR_NAME: str = "original"
    FFPP_FAKE_DIR_NAMES: Tuple[str, ...] = (
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
        "FaceShifter",
        "DeepFakeDetection",
    )

    # Frame extraction
    FRAMES_PER_VIDEO: int = 32
    FACE_DETECTOR: str = "haar" 
    FACE_MARGIN: float = 0.25
    SAVE_FULL_FRAME_IF_NO_FACE: bool = True
    IMG_SIZE: int = 380

    # Split ratio at video level
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    RANDOM_SEED: int = 42

    # Training settings
    MODEL_NAME: str = "efficientnet_b4"  
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    EPOCHS: int = 10
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    PATIENCE: int = 4
    USE_AMP: bool = True
    FAKE_THRESHOLD: float = 0.50

    # Robust training settings
    ROBUST_TRAIN_DEGRADATIONS: Tuple[str, ...] = (
        "clean",
        "jpeg",        
        "blur",        
        "downscale",   
        "noise",      
    )

    # Consistency fine-tuning settings
    CONSISTENCY_TRAIN_DEGRADATIONS: Tuple[str, ...] = (
        "jpeg",
        "blur",
        "downscale",
        "noise",
    )
    ROBUST_VAL_DEGRADATIONS: Tuple[str, ...] = (
        "clean",
        "jpeg",
        "blur",
        "downscale",
        "noise",
    )
    CONSISTENCY_LAMBDA: float = 0.5
    CONSISTENCY_LR: float = 5e-5

    # Deepfake-specific classifier head settings
    DEEP_HEAD_HIDDEN: int = 512
    DEEP_HEAD_DROPOUT1: float = 0.40
    DEEP_HEAD_DROPOUT2: float = 0.30

    CLASS_NAMES: Tuple[str, str] = ("real", "fake")


cfg = CFG()


def apply_cli_to_cfg(args):
    cfg.DATA_ROOT = args.data_root
    cfg.OUTPUT_ROOT = args.output_root
    cfg.DATA_FORMAT = args.data_format
    cfg.FRAMES_PER_VIDEO = args.frames_per_video
    cfg.FACE_DETECTOR = args.face_detector
    cfg.IMG_SIZE = args.img_size
    cfg.MODEL_NAME = args.model
    cfg.BATCH_SIZE = args.batch_size
    cfg.EPOCHS = args.epochs
    cfg.NUM_WORKERS = args.num_workers

    if hasattr(args, "lr"):
        cfg.LR = args.lr
    if hasattr(args, "weight_decay"):
        cfg.WEIGHT_DECAY = args.weight_decay
    if hasattr(args, "patience"):
        cfg.PATIENCE = args.patience
    if hasattr(args, "consistency_lambda"):
        cfg.CONSISTENCY_LAMBDA = args.consistency_lambda
    if hasattr(args, "consistency_lr"):
        cfg.CONSISTENCY_LR = args.consistency_lr

    if args.fake_dirs:
        cfg.FFPP_FAKE_DIR_NAMES = tuple([x.strip() for x in args.fake_dirs.split(",") if x.strip()])


def get_paths():
    output_root = Path(cfg.OUTPUT_ROOT)
    dataset_tag = Path(cfg.DATA_ROOT).name
    preprocessed_root = output_root / "preprocessed_image" / dataset_tag
    split_csv = output_root / f"video_split_{dataset_tag}.csv"
    frame_csv = output_root / f"frames_metadata_{dataset_tag}.csv"
    model_dir = output_root / "models"
    result_dir = output_root / "results"
    gradcam_dir = output_root / "gradcam"

    for p in [output_root, preprocessed_root, model_dir, result_dir, gradcam_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": output_root,
        "dataset_tag": dataset_tag,
        "preprocessed_root": preprocessed_root,
        "split_csv": split_csv,
        "frame_csv": frame_csv,
        "model_dir": model_dir,
        "result_dir": result_dir,
        "gradcam_dir": gradcam_dir,
    }


# 2. Utility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}


def sample_frame_indices(total_frames: int, n_samples: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= n_samples:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, n_samples, dtype=int).tolist()


# 3. Collect Videos and Split at Video Level
def collect_standard_video_paths(data_root: str) -> pd.DataFrame:
    data_root = Path(data_root)
    real_dir = data_root / cfg.REAL_DIR_NAME
    fake_dir = data_root / cfg.FAKE_DIR_NAME

    rows = []
    for label_name, label_id, folder, method in [
        ("real", 0, real_dir, "real"),
        ("fake", 1, fake_dir, "fake"),
    ]:
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        for vp in sorted(folder.rglob("*")):
            if vp.is_file() and is_video_file(vp):
                rows.append({
                    "video_path": str(vp),
                    "video_name": vp.stem,
                    "label": label_id,
                    "label_name": label_name,
                    "method": method,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No videos found. Check DATA_ROOT, real folder, and fake folder.")
    return df


def collect_ffpp_video_paths(data_root: str) -> pd.DataFrame:
    """Collect FaceForensics++ style data safely.

    IMPORTANT:
    - original is label 0, real.
    - configured fake folders are label 1, fake.
    - This prevents original videos from being accidentally saved as fake.
    """
    data_root = Path(data_root)
    rows = []

    real_dir = data_root / cfg.FFPP_REAL_DIR_NAME
    if not real_dir.exists():
        raise FileNotFoundError(f"FF++ real/original folder not found: {real_dir}")

    for vp in sorted(real_dir.rglob("*")):
        if vp.is_file() and is_video_file(vp):
            rows.append({
                "video_path": str(vp),
                "video_name": vp.stem,
                "label": 0,
                "label_name": "real",
                "method": "original",
            })

    for fake_method in cfg.FFPP_FAKE_DIR_NAMES:
        fake_dir = data_root / fake_method
        if not fake_dir.exists():
            print(f"[Warning] Fake folder not found, skipped: {fake_dir}")
            continue

        for vp in sorted(fake_dir.rglob("*")):
            if vp.is_file() and is_video_file(vp):
                rows.append({
                    "video_path": str(vp),
                    "video_name": vp.stem,
                    "label": 1,
                    "label_name": "fake",
                    "method": fake_method,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No videos found in FF++ dataset.")

    print("Collected videos by label and method:")
    print(df.groupby(["label_name", "method"]).size())
    return df


def collect_video_paths(data_root: str) -> pd.DataFrame:
    if cfg.DATA_FORMAT.lower() == "ffpp":
        return collect_ffpp_video_paths(data_root)
    if cfg.DATA_FORMAT.lower() == "standard":
        return collect_standard_video_paths(data_root)
    raise ValueError("DATA_FORMAT must be 'ffpp' or 'standard'.")


def make_video_level_split(df: pd.DataFrame) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - cfg.TRAIN_RATIO),
        random_state=cfg.RANDOM_SEED,
        stratify=df["label"],
    )

    val_size_adjusted = cfg.VAL_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_adjusted),
        random_state=cfg.RANDOM_SEED,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df = split_df.sample(frac=1, random_state=cfg.RANDOM_SEED).reset_index(drop=True)

    print("Video split summary:")
    print(split_df.groupby(["split", "label_name"]).size())
    return split_df


# 4. Face Cropping and Frame Extraction
class FaceCropper:
    def __init__(self, detector_type: str = "haar"):
        self.detector_type = detector_type.lower()
        if self.detector_type == "haar":
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            print("Using OpenCV Haar face detector.")
        elif self.detector_type == "none":
            self.haar = None
            print("No face detector. Full frames will be used.")
        else:
            raise ValueError("FACE_DETECTOR must be 'haar' or 'none'.")

    def _add_margin(self, box, w, h, margin: float):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * margin, bh * margin
        x1 = max(0, int(x1 - mx)); y1 = max(0, int(y1 - my))
        x2 = min(w, int(x2 + mx)); y2 = min(h, int(y2 + my))
        return x1, y1, x2, y2

    def center_crop_square(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        side = min(h, w)
        y0 = max((h - side) // 2, 0)
        x0 = max((w - side) // 2, 0)
        return frame_bgr[y0:y0 + side, x0:x0 + side]

    def crop_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        if self.detector_type == "none":
            return self.center_crop_square(frame_bgr)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            x1, y1, x2, y2 = self._add_margin([x, y, x + fw, y + fh], w, h, cfg.FACE_MARGIN)
            return frame_bgr[y1:y2, x1:x2]

        if cfg.SAVE_FULL_FRAME_IF_NO_FACE:
            return self.center_crop_square(frame_bgr)
        return None


def extract_frames_from_video(row: pd.Series, out_root: Path, cropper: FaceCropper) -> List[Dict]:
    video_path = Path(row["video_path"])
    split = row["split"]
    label = int(row["label"])
    label_name = row["label_name"]
    method = row["method"]
    video_stem = video_path.stem

    out_dir = out_root / split / label_name / method / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Warning] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    indices = sample_frame_indices(total_frames, cfg.FRAMES_PER_VIDEO)
    records = []

    for i, frame_idx in enumerate(indices):
        out_path = out_dir / f"frame_{i:04d}.jpg"
        if out_path.exists():
            records.append({
                "image_path": str(out_path),
                "video_path": str(video_path),
                "video_name": video_stem,
                "method": method,
                "frame_index": int(frame_idx),
                "timestamp_sec": float(frame_idx / fps),
                "label": label,
                "label_name": label_name,
                "split": split,
            })
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        face = cropper.crop_face(frame)
        if face is None or face.size == 0:
            continue

        face = cv2.resize(face, (cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        records.append({
            "image_path": str(out_path),
            "video_path": str(video_path),
            "video_name": video_stem,
            "method": method,
            "frame_index": int(frame_idx),
            "timestamp_sec": float(frame_idx / fps),
            "label": label,
            "label_name": label_name,
            "split": split,
        })

    cap.release()
    return records


def prepare_dataset():
    set_seed(cfg.RANDOM_SEED)
    paths = get_paths()

    videos_df = collect_video_paths(cfg.DATA_ROOT)
    split_df = make_video_level_split(videos_df)
    split_df.to_csv(paths["split_csv"], index=False)
    print("Saved split CSV:", paths["split_csv"])

    cropper = FaceCropper(cfg.FACE_DETECTOR)
    all_records = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Extracting preprocessed frames"):
        all_records.extend(extract_frames_from_video(row, paths["preprocessed_root"], cropper))

    frame_df = pd.DataFrame(all_records)
    frame_df.to_csv(paths["frame_csv"], index=False)
    print("Saved frame metadata:", paths["frame_csv"])

    print("Extracted image summary:")
    if not frame_df.empty:
        print(frame_df.groupby(["split", "label_name"]).size())
    else:
        print("No frames extracted.")

    print("Expected folders:")
    print(paths["preprocessed_root"] / "train" / "real")
    print(paths["preprocessed_root"] / "train" / "fake")
    print(paths["preprocessed_root"] / "val" / "real")
    print(paths["preprocessed_root"] / "val" / "fake")
    print(paths["preprocessed_root"] / "test" / "real")
    print(paths["preprocessed_root"] / "test" / "fake")


def check_preprocessed_dataset(frame_root: Optional[str] = None):
    paths = get_paths()
    root = Path(frame_root) if frame_root else paths["preprocessed_root"]
    print("Checking preprocessed dataset root:", root)

    rows = []
    for split in ["train", "val", "test"]:
        for label_name in ["real", "fake"]:
            folder = root / split / label_name
            count = len(list(folder.rglob("*.jpg"))) if folder.exists() else 0
            rows.append({"split": split, "label_name": label_name, "image_count": count, "folder": str(folder)})

    df = pd.DataFrame(rows)
    print(df)

    missing = df[df["image_count"] == 0]
    if len(missing) > 0:
        print("\n[Warning] Some folders have 0 images. Do not train until both real and fake exist in each split.")
    else:
        print("\nOK: real and fake images exist in train/val/test.")
    return df


# 5. Degradation Functions for Robustness
def apply_jpeg_compression(img: Image.Image, quality: int = 40) -> Image.Image:
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_blur(img: Image.Image, kernel_size: int = 7) -> Image.Image:
    arr = np.array(img)
    if kernel_size % 2 == 0:
        kernel_size += 1
    arr = cv2.GaussianBlur(arr, (kernel_size, kernel_size), 0)
    return Image.fromarray(arr)


def apply_noise(img: Image.Image, sigma: float = 12.0) -> Image.Image:
    """Add random Gaussian noise to simulate noisy real-world evidence."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_downscale(img: Image.Image, scale: float = 0.35) -> Image.Image:
    w, h = img.size
    small_w = max(16, int(w * scale)); small_h = max(16, int(h * scale))
    small = img.resize((small_w, small_h), Image.BILINEAR)
    restored = small.resize((w, h), Image.BILINEAR)
    return restored


def degrade_image(img: Image.Image, degradation: str) -> Image.Image:
    degradation = degradation.lower()
    if degradation == "clean":
        return img
    if degradation == "jpeg":
        return apply_jpeg_compression(img, quality=random.choice([30, 40, 60]))
    if degradation == "blur":
        return apply_gaussian_blur(img, kernel_size=random.choice([3, 5, 7]))
    if degradation == "downscale":
        return apply_downscale(img, scale=random.choice([0.25, 0.35, 0.50]))
    if degradation == "noise":
        return apply_noise(img, sigma=random.choice([8.0, 12.0, 18.0]))
    if degradation == "mixed":
        out = img
        for d in random.sample(["jpeg", "blur", "downscale", "noise"], k=random.choice([1, 2])):
            out = degrade_image(out, d)
        return out
    raise ValueError(f"Unknown degradation: {degradation}")


def make_fixed_degraded_copy(img: Image.Image, degradation: str, severity: str = "medium") -> Image.Image:
    if degradation == "clean":
        return img
    if degradation == "jpeg":
        return apply_jpeg_compression(img, quality={"mild": 70, "medium": 40, "severe": 20}[severity])
    if degradation == "blur":
        return apply_gaussian_blur(img, kernel_size={"mild": 3, "medium": 7, "severe": 11}[severity])
    if degradation == "downscale":
        return apply_downscale(img, scale={"mild": 0.60, "medium": 0.35, "severe": 0.20}[severity])
    if degradation == "noise":
        return apply_noise(img, sigma={"mild": 6.0, "medium": 12.0, "severe": 20.0}[severity])
    raise ValueError(f"Unknown degradation: {degradation}")


# 6. Lazy Training Imports and Training Utilities
def ensure_torch_imports():
    global torch, nn, optim, Dataset, DataLoader, transforms
    global efficientnet_b4, EfficientNet_B4_Weights
    global autocast, GradScaler, plt

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda.amp import autocast, GradScaler
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import efficientnet_b4
    from torchvision.models import EfficientNet_B4_Weights
    import matplotlib.pyplot as plt

    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepfakeFrameDatasetLazy:
    pass


def build_dataset_class():
    class DeepfakeFrameDataset(Dataset):
        def __init__(
            self,
            frame_root: Path,
            split: str,
            transform=None,
            robust_training: bool = False,
            fixed_test_degradation: str = "clean",
            fixed_severity: str = "medium",
        ):
            self.frame_root = Path(frame_root)
            self.split = split
            self.transform = transform
            self.robust_training = robust_training
            self.fixed_test_degradation = fixed_test_degradation
            self.fixed_severity = fixed_severity

            self.samples = []
            for label_name, label_id in [("real", 0), ("fake", 1)]:
                folder = self.frame_root / split / label_name
                if not folder.exists():
                    raise FileNotFoundError(f"Folder not found: {folder}")
                for img_path in sorted(folder.rglob("*.jpg")):
                    self.samples.append((str(img_path), label_id))

            if len(self.samples) == 0:
                raise ValueError(f"No images found for split={split} in {self.frame_root}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("RGB")

            if self.split == "train" and self.robust_training:
                img = degrade_image(img, random.choice(cfg.ROBUST_TRAIN_DEGRADATIONS))

            if self.split in ["val", "test"]:
                img = make_fixed_degraded_copy(img, self.fixed_test_degradation, self.fixed_severity)

            if self.transform:
                img = self.transform(img)

            return img, torch.tensor(label, dtype=torch.float32), img_path

    return DeepfakeFrameDataset


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_loader(frame_root: Path, split: str, robust_training=False, degradation="clean", severity="medium", shuffle=False):
    DeepfakeFrameDataset = build_dataset_class()
    ds = DeepfakeFrameDataset(
        frame_root=frame_root,
        split=split,
        transform=get_transforms(train=(split == "train")),
        robust_training=robust_training,
        fixed_test_degradation=degradation,
        fixed_severity=severity,
    )
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.NUM_WORKERS, pin_memory=True)


# Return clean and degraded versions of the same image for consistency learning
def build_pair_dataset_class():
    class DeepfakePairDataset(Dataset):

        def __init__(
            self,
            frame_root: Path,
            split: str,
            transform=None,
        ):
            self.frame_root = Path(frame_root)
            self.split = split
            self.transform = transform

            self.samples = []
            for label_name, label_id in [("real", 0), ("fake", 1)]:
                folder = self.frame_root / split / label_name
                if not folder.exists():
                    raise FileNotFoundError(f"Folder not found: {folder}")
                for img_path in sorted(folder.rglob("*.jpg")):
                    self.samples.append((str(img_path), label_id))

            if len(self.samples) == 0:
                raise ValueError(f"No images found for split={split} in {self.frame_root}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            clean_img = Image.open(img_path).convert("RGB")

            degradation = random.choice(cfg.CONSISTENCY_TRAIN_DEGRADATIONS)
            degraded_img = degrade_image(clean_img.copy(), degradation)

            if self.transform:
                clean_img = self.transform(clean_img)
                degraded_img = self.transform(degraded_img)

            return (
                clean_img,
                degraded_img,
                torch.tensor(label, dtype=torch.float32),
                img_path,
                degradation,
            )

    return DeepfakePairDataset


def get_pair_transforms():
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_pair_loader(frame_root: Path, split: str, shuffle=True):
    DeepfakePairDataset = build_pair_dataset_class()
    ds = DeepfakePairDataset(
        frame_root=frame_root,
        split=split,
        transform=get_pair_transforms(),
    )
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.NUM_WORKERS, pin_memory=True)


# Deep classifier head
def build_deepfake_classifier_head(in_features: int):
    return nn.Sequential(
        nn.Dropout(p=cfg.DEEP_HEAD_DROPOUT1),
        nn.Linear(in_features, cfg.DEEP_HEAD_HIDDEN),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(cfg.DEEP_HEAD_HIDDEN),
        nn.Dropout(p=cfg.DEEP_HEAD_DROPOUT2),
        nn.Linear(cfg.DEEP_HEAD_HIDDEN, 1),
    )


# Build EfficientNet with either a simple or deepfake-specific classifier head
def build_efficientnet(model_name: str = "efficientnet_b4", pretrained: bool = True, head_type: str = "simple"):
    model_name = model_name.lower()
    head_type = head_type.lower()

    if model_name == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
    else:
        raise ValueError("MODEL_NAME must be 'efficientnet_b4'.")

    if head_type == "simple":
        model.classifier[1] = nn.Linear(in_features, 1)
    elif head_type == "deep":
        model.classifier = build_deepfake_classifier_head(in_features)
    else:
        raise ValueError("head_type must be 'simple' or 'deep'.")

    return model

def binary_metrics(y_true, y_prob, threshold=0.5) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc, "confusion_matrix": cm.tolist()}


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_probs, all_labels = [], []

    for imgs, labels, _ in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)

        if cfg.USE_AMP and scaler is not None and device.type == "cuda":
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().reshape(-1).tolist())

    metrics = binary_metrics(all_labels, all_probs, threshold=cfg.FAKE_THRESHOLD)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def evaluate_model(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_probs, all_labels, all_paths = [], [], []

    for imgs, labels, paths in tqdm(loader, desc="Evaluating", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        logits = model(imgs)
        if criterion is not None:
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
        all_paths.extend(paths)

    metrics = binary_metrics(all_labels, all_probs, threshold=threshold)
    metrics["loss"] = running_loss / len(loader.dataset) if criterion is not None else np.nan
    pred_df = pd.DataFrame({
        "image_path": all_paths,
        "label": all_labels,
        "fake_probability": all_probs,
        "prediction": (np.asarray(all_probs) >= threshold).astype(int),
    })
    return metrics, pred_df


def train_model(experiment_name: str, robust_training: bool = False):
    device = ensure_torch_imports()
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    paths = get_paths()
    check_preprocessed_dataset(str(paths["preprocessed_root"]))

    model = build_efficientnet(cfg.MODEL_NAME, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    scaler = GradScaler(enabled=(cfg.USE_AMP and device.type == "cuda"))

    train_loader = make_loader(paths["preprocessed_root"], "train", robust_training=robust_training, shuffle=True)
    val_loader = make_loader(paths["preprocessed_root"], "val", degradation="clean")

    best_f1 = -1
    best_path = paths["model_dir"] / f"{experiment_name}_best.pt"
    history = []
    bad_epochs = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics, _ = evaluate_model(model, val_loader, criterion, device, threshold=cfg.FAKE_THRESHOLD)
        scheduler.step(val_metrics["f1"])

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items() if k != "confusion_matrix"},
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "cfg": cfg.__dict__, "best_f1": best_f1, "epoch": epoch, "head_type": "simple"}, best_path)
            print("Saved best model:", best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.PATIENCE:
                print("Early stopping triggered.")
                break

    hist_df = pd.DataFrame(history)
    hist_path = paths["result_dir"] / f"{experiment_name}_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print("Saved history:", hist_path)
    return best_path



def load_matching_weights(model, checkpoint_path: str, device, exclude_classifier: bool = True):
    checkpoint_path = str(checkpoint_path)
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"[Warning] Baseline checkpoint not found, training from ImageNet weights instead: {checkpoint_path}")
        return model

    ckpt = torch.load(checkpoint_path, map_location=device)
    source_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    target_state = model.state_dict()

    compatible = {}
    skipped = []

    for k, v in source_state.items():
        if exclude_classifier and k.startswith("classifier"):
            skipped.append(k)
            continue

        if k in target_state and target_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Loaded compatible weights from: {checkpoint_path}")
    print(f"Loaded tensors: {len(compatible)}")
    print(f"Skipped tensors: {len(skipped)}")
    print(f"Missing tensors after partial load: {len(missing)}")
    print(f"Unexpected tensors after partial load: {len(unexpected)}")
    return model

# Train with clean + degraded image pairs & consistency loss
def train_consistency_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_clean_loss = 0.0
    running_degraded_loss = 0.0
    running_consistency_loss = 0.0
    all_probs, all_labels = [], []

    for clean_imgs, degraded_imgs, labels, _, _ in tqdm(loader, desc="Consistency training", leave=False):
        clean_imgs = clean_imgs.to(device, non_blocking=True)
        degraded_imgs = degraded_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)

        if cfg.USE_AMP and scaler is not None and device.type == "cuda":
            with autocast():
                clean_logits = model(clean_imgs)
                degraded_logits = model(degraded_imgs)

                clean_loss = criterion(clean_logits, labels)
                degraded_loss = criterion(degraded_logits, labels)

                clean_prob = torch.sigmoid(clean_logits)
                degraded_prob = torch.sigmoid(degraded_logits)
                consistency_loss = nn.functional.mse_loss(clean_prob, degraded_prob)

                loss = clean_loss + degraded_loss + cfg.CONSISTENCY_LAMBDA * consistency_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            clean_logits = model(clean_imgs)
            degraded_logits = model(degraded_imgs)

            clean_loss = criterion(clean_logits, labels)
            degraded_loss = criterion(degraded_logits, labels)

            clean_prob = torch.sigmoid(clean_logits)
            degraded_prob = torch.sigmoid(degraded_logits)
            consistency_loss = nn.functional.mse_loss(clean_prob, degraded_prob)

            loss = clean_loss + degraded_loss + cfg.CONSISTENCY_LAMBDA * consistency_loss
            loss.backward()
            optimizer.step()

        batch_size = clean_imgs.size(0)
        running_loss += loss.item() * batch_size
        running_clean_loss += clean_loss.item() * batch_size
        running_degraded_loss += degraded_loss.item() * batch_size
        running_consistency_loss += consistency_loss.item() * batch_size

        avg_prob = ((torch.sigmoid(clean_logits) + torch.sigmoid(degraded_logits)) / 2.0).detach().cpu().numpy().reshape(-1)
        all_probs.extend(avg_prob.tolist())
        all_labels.extend(labels.detach().cpu().numpy().reshape(-1).tolist())

    metrics = binary_metrics(all_labels, all_probs, threshold=cfg.FAKE_THRESHOLD)
    n = len(loader.dataset)
    metrics["loss"] = running_loss / n
    metrics["clean_cls_loss"] = running_clean_loss / n
    metrics["degraded_cls_loss"] = running_degraded_loss / n
    metrics["consistency_loss"] = running_consistency_loss / n
    return metrics


def evaluate_validation_robust_score(model, frame_root: Path, criterion, device):
    rows = []
    for deg in cfg.ROBUST_VAL_DEGRADATIONS:
        loader = make_loader(frame_root, "val", degradation=deg, severity="medium", shuffle=False)
        metrics, _ = evaluate_model(model, loader, criterion, device, threshold=cfg.FAKE_THRESHOLD)
        rows.append({
            "degradation": deg,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "loss": metrics["loss"],
        })

    val_df = pd.DataFrame(rows)
    robust_score = float(val_df["f1"].mean())
    mean_auc = float(val_df["auc"].mean())
    mean_acc = float(val_df["accuracy"].mean())
    return robust_score, mean_auc, mean_acc, val_df


def train_consistency_model(experiment_name: str, baseline_checkpoint: str = ""):
    device = ensure_torch_imports()
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    paths = get_paths()
    check_preprocessed_dataset(str(paths["preprocessed_root"]))

    if not baseline_checkpoint:
        baseline_checkpoint = str(paths["model_dir"] / f"baseline_{cfg.MODEL_NAME}_clean_best.pt")

    print("Baseline checkpoint for fine-tuning:", baseline_checkpoint)
    print("Consistency lambda:", cfg.CONSISTENCY_LAMBDA)
    print("Consistency LR:", cfg.CONSISTENCY_LR)
    print("Train degradations:", cfg.CONSISTENCY_TRAIN_DEGRADATIONS)
    print("Validation degradations:", cfg.ROBUST_VAL_DEGRADATIONS)

    model = build_efficientnet(cfg.MODEL_NAME, pretrained=True, head_type="deep").to(device)
    model = load_matching_weights(model, baseline_checkpoint, device, exclude_classifier=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.CONSISTENCY_LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    scaler = GradScaler(enabled=(cfg.USE_AMP and device.type == "cuda"))

    train_loader = make_pair_loader(paths["preprocessed_root"], "train", shuffle=True)

    best_score = -1
    best_path = paths["model_dir"] / f"{experiment_name}_best.pt"
    history = []
    bad_epochs = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")
        train_metrics = train_consistency_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        robust_val_f1, robust_val_auc, robust_val_acc, val_df = evaluate_validation_robust_score(
            model, paths["preprocessed_root"], criterion, device
        )

        scheduler.step(robust_val_f1)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items() if k != "confusion_matrix"},
            "val_robust_mean_f1": robust_val_f1,
            "val_robust_mean_auc": robust_val_auc,
            "val_robust_mean_accuracy": robust_val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }

        for _, r in val_df.iterrows():
            deg = r["degradation"]
            row[f"val_{deg}_f1"] = r["f1"]
            row[f"val_{deg}_auc"] = r["auc"]
            row[f"val_{deg}_accuracy"] = r["accuracy"]
            row[f"val_{deg}_loss"] = r["loss"]

        history.append(row)

        print(
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Robust Val Mean F1: {robust_val_f1:.4f} | "
            f"Robust Val Mean AUC: {robust_val_auc:.4f}"
        )
        print(val_df[["degradation", "accuracy", "f1", "auc", "loss"]])

        if robust_val_f1 > best_score:
            best_score = robust_val_f1
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "best_robust_val_f1": best_score,
                "epoch": epoch,
                "head_type": "deep",
                "method": "consistency_finetuned_efficientnet",
                "baseline_checkpoint": baseline_checkpoint,
            }, best_path)
            print("Saved best consistency model:", best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.PATIENCE:
                print("Early stopping triggered.")
                break

    hist_df = pd.DataFrame(history)
    hist_path = paths["result_dir"] / f"{experiment_name}_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print("Saved consistency training history:", hist_path)
    return best_path


def load_trained_model(model_path: str, device, head_type: Optional[str] = None):
    ckpt = torch.load(model_path, map_location=device)
    if head_type is None:
        head_type = ckpt.get("head_type", "simple") if isinstance(ckpt, dict) else "simple"
    model = build_efficientnet(cfg.MODEL_NAME, pretrained=False, head_type=head_type).to(device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_across_degradations(model_path: str, experiment_name: str, frame_root: Optional[str] = None, dataset_tag: str = "within_domain", severity: str = "medium", head_type: Optional[str] = None):
    device = ensure_torch_imports()
    paths = get_paths()
    frame_root_path = Path(frame_root) if frame_root else paths["preprocessed_root"]
    model = load_trained_model(model_path, device, head_type=head_type)
    criterion = nn.BCEWithLogitsLoss()
    rows = []

    for deg in ["clean", "jpeg", "blur", "downscale", "noise"]:
        print(f"Evaluating {experiment_name} on {dataset_tag} / {deg}...")
        loader = make_loader(frame_root_path, "test", degradation=deg, severity=severity)
        metrics, pred_df = evaluate_model(model, loader, criterion, device, threshold=cfg.FAKE_THRESHOLD)
        pred_path = paths["result_dir"] / f"{experiment_name}_{dataset_tag}_predictions_{deg}_{severity}.csv"
        pred_df.to_csv(pred_path, index=False)

        rows.append({
            "experiment": experiment_name,
            "dataset_tag": dataset_tag,
            "degradation": deg,
            "severity": severity,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "loss": metrics["loss"],
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        })

    result_df = pd.DataFrame(rows)
    clean_f1 = float(result_df.loc[result_df["degradation"] == "clean", "f1"].iloc[0])
    clean_acc = float(result_df.loc[result_df["degradation"] == "clean", "accuracy"].iloc[0])
    result_df["f1_drop"] = clean_f1 - result_df["f1"]
    result_df["accuracy_drop"] = clean_acc - result_df["accuracy"]

    out_path = paths["result_dir"] / f"{experiment_name}_{dataset_tag}_degradation_results_{severity}.csv"
    result_df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(result_df)
    return result_df


def compare_results(baseline_csv: str, robust_csv: str, output_name: str = "baseline_vs_robust_comparison.csv"):
    paths = get_paths()
    baseline_df = pd.read_csv(baseline_csv)
    robust_df = pd.read_csv(robust_csv)
    merged = baseline_df.merge(robust_df, on=["dataset_tag", "degradation", "severity"], suffixes=("_baseline", "_robust"))
    merged["f1_improvement"] = merged["f1_robust"] - merged["f1_baseline"]
    merged["accuracy_improvement"] = merged["accuracy_robust"] - merged["accuracy_baseline"]
    merged["auc_improvement"] = merged["auc_robust"] - merged["auc_baseline"]
    out_path = paths["result_dir"] / output_name
    merged.to_csv(out_path, index=False)
    print("Saved comparison:", out_path)
    print(merged[["dataset_tag", "degradation", "accuracy_baseline", "accuracy_robust", "f1_baseline", "f1_robust", "auc_baseline", "auc_robust", "auc_improvement"]])


# 7. Main CLI
def build_parser():
    parser = argparse.ArgumentParser(description="Fixed FaceForensics++ robust EfficientNet pipeline")
    parser.add_argument("mode", choices=["prepare", "check", "train_baseline", "train_robust", "train_consistency", "eval_baseline", "eval_robust", "eval_consistency", "full"], help="Pipeline mode")
    parser.add_argument("--data_root", type=str, default=cfg.DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=cfg.OUTPUT_ROOT)
    parser.add_argument("--data_format", type=str, default=cfg.DATA_FORMAT, choices=["ffpp", "standard"])
    parser.add_argument("--fake_dirs", type=str, default="", help="Comma-separated FF++ fake folders. Leave empty for default.")
    parser.add_argument("--frames_per_video", type=int, default=cfg.FRAMES_PER_VIDEO)
    parser.add_argument("--face_detector", type=str, default=cfg.FACE_DETECTOR, choices=["haar", "none"])
    parser.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    parser.add_argument("--model", type=str, default=cfg.MODEL_NAME, choices=["efficientnet_b4"])
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--consistency_lambda", type=float, default=cfg.CONSISTENCY_LAMBDA)
    parser.add_argument("--consistency_lr", type=float, default=cfg.CONSISTENCY_LR)
    parser.add_argument("--baseline_checkpoint", type=str, default="", help="Baseline checkpoint used to initialize consistency fine-tuning.")
    parser.add_argument("--model_path", type=str, default="", help="Model path for eval mode")
    parser.add_argument("--external_frame_root", type=str, default="", help="Optional external preprocessed dataset for cross-domain evaluation")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    apply_cli_to_cfg(args)
    set_seed(cfg.RANDOM_SEED)

    paths = get_paths()
    print("DATA_ROOT:", cfg.DATA_ROOT)
    print("OUTPUT_ROOT:", cfg.OUTPUT_ROOT)
    print("PREPROCESSED_ROOT:", paths["preprocessed_root"])
    print("DATA_FORMAT:", cfg.DATA_FORMAT)

    if args.mode == "prepare":
        prepare_dataset()
        check_preprocessed_dataset()

    elif args.mode == "check":
        check_preprocessed_dataset()

    elif args.mode == "train_baseline":
        train_model(f"baseline_{cfg.MODEL_NAME}_clean", robust_training=False)

    elif args.mode == "train_robust":
        train_model(f"da_{cfg.MODEL_NAME}_robust", robust_training=True)

    elif args.mode == "train_consistency":
        baseline_checkpoint = args.baseline_checkpoint
        if not baseline_checkpoint:
            baseline_checkpoint = str(paths["model_dir"] / f"baseline_{cfg.MODEL_NAME}_clean_best.pt")
        train_consistency_model(f"consistency_{cfg.MODEL_NAME}_robust", baseline_checkpoint=baseline_checkpoint)

    elif args.mode == "eval_baseline":
        if not args.model_path:
            args.model_path = str(paths["model_dir"] / f"baseline_{cfg.MODEL_NAME}_clean_best.pt")
        evaluate_across_degradations(args.model_path, f"baseline_{cfg.MODEL_NAME}_clean", dataset_tag="within_domain", head_type="simple")
        if args.external_frame_root:
            evaluate_across_degradations(args.model_path, f"baseline_{cfg.MODEL_NAME}_clean", frame_root=args.external_frame_root, dataset_tag="cross_domain", head_type="simple")

    elif args.mode == "eval_robust":
        if not args.model_path:
            args.model_path = str(paths["model_dir"] / f"da_{cfg.MODEL_NAME}_robust_best.pt")
        evaluate_across_degradations(args.model_path, f"da_{cfg.MODEL_NAME}_robust", dataset_tag="within_domain", head_type="simple")
        if args.external_frame_root:
            evaluate_across_degradations(args.model_path, f"da_{cfg.MODEL_NAME}_robust", frame_root=args.external_frame_root, dataset_tag="cross_domain", head_type="simple")

    elif args.mode == "eval_consistency":
        if not args.model_path:
            args.model_path = str(paths["model_dir"] / f"consistency_{cfg.MODEL_NAME}_robust_best.pt")
        evaluate_across_degradations(args.model_path, f"consistency_{cfg.MODEL_NAME}_robust", dataset_tag="within_domain", head_type="deep")
        if args.external_frame_root:
            evaluate_across_degradations(args.model_path, f"consistency_{cfg.MODEL_NAME}_robust", frame_root=args.external_frame_root, dataset_tag="cross_domain", head_type="deep")

    elif args.mode == "full":
        prepare_dataset()
        check_preprocessed_dataset()
        baseline_path = train_model(f"baseline_{cfg.MODEL_NAME}_clean", robust_training=False)
        robust_path = train_model(f"da_{cfg.MODEL_NAME}_robust", robust_training=True)
        evaluate_across_degradations(str(baseline_path), f"baseline_{cfg.MODEL_NAME}_clean", dataset_tag="within_domain")
        evaluate_across_degradations(str(robust_path), f"da_{cfg.MODEL_NAME}_robust", dataset_tag="within_domain")


if __name__ == "__main__":
    main()
