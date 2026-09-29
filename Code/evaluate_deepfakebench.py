# Evaluate DeepfakeBench / Yan et al. EfficientNet-B4 or Xception
# Reference: https://github.com/SCLBD/DeepfakeBench/tree/main (Yan et.al)

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# 1. Degradation functions, matched to the project evaluation idea
def apply_jpeg_compression(img: Image.Image, quality: int = 40) -> Image.Image:
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
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_downscale(img: Image.Image, scale: float = 0.35) -> Image.Image:
    w, h = img.size
    small_w = max(16, int(w * scale))
    small_h = max(16, int(h * scale))
    small = img.resize((small_w, small_h), Image.BILINEAR)
    restored = small.resize((w, h), Image.BILINEAR)
    return restored


def make_fixed_degraded_copy(img: Image.Image, degradation: str, severity: str = "medium") -> Image.Image:
    degradation = degradation.lower()
    severity = severity.lower()

    if degradation == "clean":
        return img

    if degradation == "jpeg":
        quality = {"mild": 70, "medium": 40, "severe": 20}[severity]
        return apply_jpeg_compression(img, quality=quality)

    if degradation == "blur":
        kernel = {"mild": 3, "medium": 7, "severe": 11}[severity]
        return apply_gaussian_blur(img, kernel_size=kernel)

    if degradation == "downscale":
        scale = {"mild": 0.60, "medium": 0.35, "severe": 0.20}[severity]
        return apply_downscale(img, scale=scale)

    if degradation == "noise":
        sigma = {"mild": 6.0, "medium": 12.0, "severe": 20.0}[severity]
        return apply_noise(img, sigma=sigma)

    raise ValueError(f"Unknown degradation: {degradation}")



# 2. Dataset
def resolve_split_root(image_root: Path, split: str) -> Path:
    if split is None:
        split = ""
    split = split.strip()
    if split.lower() in {"", "none", "external", "root"}:
        return image_root
    if (image_root / split).exists():
        return image_root / split
    return image_root


def collect_image_samples(image_root: Path, split: str) -> pd.DataFrame:
    split_root = resolve_split_root(image_root, split)

    rows = []
    for label_name, label_id in [("real", 0), ("fake", 1)]:
        class_dir = split_root / label_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {class_dir}\n"
                f"Use --image_root as either the dataset root containing split folders, "
                f"or the split folder containing real/ and fake/."
            )

        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                parts = p.parts
                video_name = p.parent.name
                method = p.parent.parent.name if len(p.parents) >= 2 else "unknown"

                rows.append({
                    "image_path": str(p),
                    "label": label_id,
                    "label_name": label_name,
                    "method": method,
                    "video_name": video_name,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No image files found under: {split_root}")
    return df


class ImageFolderFrameDataset(Dataset):
    def __init__(
        self,
        samples_df: pd.DataFrame,
        resolution: int,
        mean: List[float],
        std: List[float],
        degradation: str = "clean",
        severity: str = "medium",
    ):
        self.samples_df = samples_df.reset_index(drop=True)
        self.degradation = degradation
        self.severity = severity
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        row = self.samples_df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = make_fixed_degraded_copy(img, self.degradation, self.severity)
        img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "image_path": row["image_path"],
            "video_name": row["video_name"],
            "method": row["method"],
        }


# 3. DeepfakeBench model loading
def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DeepfakeBenchDetectorAdapter(nn.Module):

    def __init__(self, detector: str, config: Dict, deepfakebench_root: Path, device: torch.device):
        super().__init__()
        self.detector = detector.lower()
        self.config = config
        self.deepfakebench_root = deepfakebench_root
        self.device = device

        training_root = deepfakebench_root / "training"
        if str(training_root) not in sys.path:
            sys.path.insert(0, str(training_root))

        # Load the original network files from DeepfakeBench
        if self.detector == "efficientnetb4":
            module = import_module_from_path(
                "dfb_original_efficientnetb4",
                training_root / "networks" / "efficientnetb4.py",
            )
            backbone_config = dict(config["backbone_config"])
            pretrained = config.get("pretrained", None)
            backbone_config["pretrained"] = self._resolve_optional_path(pretrained)
            if backbone_config["pretrained"] is None:
                backbone_config["pretrained"] = False
            self.backbone = module.EfficientNetB4(backbone_config)

        elif self.detector == "xception":
            module = import_module_from_path(
                "dfb_original_xception",
                training_root / "networks" / "xception.py",
            )
            backbone_config = dict(config["backbone_config"])
            self.backbone = module.Xception(backbone_config)

            # Replicate DeepfakeBench XceptionDetector's optional pretrained loading.
            pretrained = self._resolve_optional_path(config.get("pretrained", None))
            if pretrained is not None and Path(pretrained).exists():
                state_dict = torch.load(pretrained, map_location=device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                fixed_state = {}
                for name, weights in state_dict.items():
                    if name.startswith("module."):
                        name = name[len("module."):]
                    if "pointwise" in name and getattr(weights, "ndim", 0) == 2:
                        weights = weights.unsqueeze(-1).unsqueeze(-1)
                    if "fc" not in name:
                        fixed_state[name] = weights
                self.backbone.load_state_dict(fixed_state, strict=False)
                print(f"Loaded Xception backbone pretrained weights: {pretrained}")
            else:
                print(
                    "[Warning] Xception pretrained backbone file not found or disabled. "
                    "This is acceptable only if --weights_path is a full trained detector checkpoint."
                )
        else:
            raise ValueError("--detector must be efficientnetb4 or xception")

    def _resolve_optional_path(self, value):
        if value is None:
            return None
        if isinstance(value, str) and value.lower() in {"none", "null", "false", ""}:
            return None
        p = Path(value)
        if not p.is_absolute():
            p = self.deepfakebench_root / p
        return str(p)

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = self.backbone.features(data_dict["image"])
        logits = self.backbone.classifier(features)
        prob = torch.softmax(logits, dim=1)[:, 1]
        return {"cls": logits, "prob": prob, "feat": features}


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "network",
            "detector",
        ]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        # Common wrappers
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out


def load_detector_checkpoint(
    model: nn.Module,
    weights_path: Optional[str],
    device: torch.device,
    strict: bool = True,
):
    if not weights_path:
        print("[Warning] No --weights_path provided. Evaluating with backbone/pretrained initialization only.")
        print("[Warning] This is NOT a trained Yan et al. detector result.")
        return

    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    ckpt = torch.load(str(p), map_location=device)
    state_dict = normalize_state_dict_keys(extract_state_dict(ckpt))

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        print(f"Loaded checkpoint: {p}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    except RuntimeError as e:
        print("[Warning] Strict checkpoint loading failed.")
        print(str(e))
        print("[Info] Retrying with strict=False.")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint with strict=False: {p}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")


# 4. Metrics and evaluation
def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_one_condition(
    model: nn.Module,
    samples_df: pd.DataFrame,
    config: Dict,
    degradation: str,
    severity: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    threshold: float,
) -> Tuple[Dict, pd.DataFrame]:
    ds = ImageFolderFrameDataset(
        samples_df=samples_df,
        resolution=int(config.get("resolution", 256)),
        mean=list(config.get("mean", [0.5, 0.5, 0.5])),
        std=list(config.get("std", [0.5, 0.5, 0.5])),
        degradation=degradation,
        severity=severity,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    all_probs, all_labels, all_paths, all_videos, all_methods = [], [], [], [], []

    for batch in tqdm(loader, desc=f"Evaluating {degradation}", leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        pred = model({"image": imgs, "label": labels})
        probs = pred["prob"].detach().cpu().numpy().reshape(-1)

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
        all_paths.extend(batch["image_path"])
        all_videos.extend(batch["video_name"])
        all_methods.extend(batch["method"])

    pred_df = pd.DataFrame({
        "image_path": all_paths,
        "video_name": all_videos,
        "method": all_methods,
        "label": all_labels,
        "fake_probability": all_probs,
        "prediction": (np.asarray(all_probs) >= threshold).astype(int),
        "degradation": degradation,
        "severity": severity,
    })

    frame_metrics = binary_metrics(all_labels, all_probs, threshold=threshold)

    return frame_metrics, pred_df


# Main
def build_argparser():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepfakeBench EfficientNet-B4/Xception on this project's preprocessed image folders."
    )

    parser.add_argument("--deepfakebench_root", type=str, required=True,
                        help="Path to DeepfakeBench-main folder.")
    parser.add_argument("--detector", type=str, required=True, choices=["efficientnetb4", "xception"],
                        help="DeepfakeBench detector to evaluate.")
    parser.add_argument("--detector_yaml", type=str, default="",
                        help="Optional detector YAML path. If omitted, uses DeepfakeBench training/config/detector/{detector}.yaml.")
    parser.add_argument("--weights_path", type=str, default="",
                        help="Path to trained detector checkpoint. Strongly recommended.")
    parser.add_argument("--pretrained_backbone", type=str, default="",
                        help="Optional override for config['pretrained'] backbone path. Use 'None' to disable.")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Project preprocessed root, either dataset root or split root.")
    parser.add_argument("--split", type=str, default="test",
                        help="Split to evaluate, e.g. test. Use empty string if image_root already points to a split folder.")
    parser.add_argument("--dataset_tag", type=str, default="my_dataset",
                        help="Name used in output CSV.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Folder to save prediction/result CSV files.")
    parser.add_argument("--degradations", nargs="+",
                        default=["clean"],
                        choices=["clean", "jpeg", "blur", "downscale", "noise"],
                        help="Test degradations to evaluate.")
    parser.add_argument("--severity", type=str, default="medium",
                        choices=["mild", "medium", "severe"],
                        help="Fixed degradation severity.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, cuda, or cpu.")
    parser.add_argument("--strict_load", action="store_true",
                        help="Use strict checkpoint loading first. If it fails, script retries strict=False.")

    return parser


def main():
    args = build_argparser().parse_args()

    deepfakebench_root = Path(args.deepfakebench_root).resolve()
    if not deepfakebench_root.exists():
        raise FileNotFoundError(f"DeepfakeBench root not found: {deepfakebench_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)

    if args.detector_yaml:
        detector_yaml = Path(args.detector_yaml)
    else:
        detector_yaml = deepfakebench_root / "training" / "config" / "detector" / f"{args.detector}.yaml"

    if not detector_yaml.exists():
        raise FileNotFoundError(f"Detector YAML not found: {detector_yaml}")

    config = load_yaml(detector_yaml)

    if args.pretrained_backbone:
        if args.pretrained_backbone.lower() == "none":
            config["pretrained"] = "None"
        else:
            config["pretrained"] = args.pretrained_backbone

    # If config uses relative pretrained path, the adapter resolves it relative to DeepfakeBench root.
    model = DeepfakeBenchDetectorAdapter(
        detector=args.detector,
        config=config,
        deepfakebench_root=deepfakebench_root,
        device=device,
    ).to(device)

    load_detector_checkpoint(
        model=model,
        weights_path=args.weights_path,
        device=device,
        strict=args.strict_load,
    )

    image_root = Path(args.image_root).resolve()
    samples_df = collect_image_samples(image_root=image_root, split=args.split)
    print("Collected images:")
    print(samples_df.groupby(["label_name"]).size())
    print("Total images:", len(samples_df))

    summary_rows = []

    for degradation in args.degradations:
        frame_metrics, pred_df = evaluate_one_condition(
            model=model,
            samples_df=samples_df,
            config=config,
            degradation=degradation,
            severity=args.severity,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            threshold=args.threshold,
        )

        pred_path = output_dir / f"{args.detector}_{args.dataset_tag}_frame_predictions_{degradation}_{args.severity}.csv"
        pred_df.to_csv(pred_path, index=False)

        summary_row = {
            "detector": args.detector,
            "dataset_tag": args.dataset_tag,
            "level": "frame",
            "degradation": degradation,
            "severity": args.severity,
            **{k: v for k, v in frame_metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": json.dumps(frame_metrics["confusion_matrix"]),
            "num_samples": len(pred_df),
            "prediction_csv": str(pred_path),
        }
        summary_rows.append(summary_row)

        print(f"\n{args.detector} | {args.dataset_tag} | {degradation} | frame-level")
        print({k: v for k, v in frame_metrics.items() if k != "confusion_matrix"})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{args.detector}_{args.dataset_tag}_summary_{args.severity}.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nSaved summary:", summary_path)
    print(summary_df)


if __name__ == "__main__":
    main()
