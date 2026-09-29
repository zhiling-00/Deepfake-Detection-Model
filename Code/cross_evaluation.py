# Dataset Celeb-DF-v2: https://www.kaggle.com/datasets/pranabr0y/celebdf-v2image-dataset
# Dataset DFDC: from "dfdc_video_preprocess.py"

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import deepfake_detection_efficientnet as dp


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_test_folder(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "test",
        dataset_root / "Test",
        dataset_root / "TEST",
    ]

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    raise FileNotFoundError(
        f"Cannot find test folder under {dataset_root}. Expected one of: test, Test, TEST"
    )


def collect_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def convert_to_jpg_compatible_root(
    dataset_root: Path,
    jpg_root: Path,
    force_rebuild: bool = False,
) -> Path:
    test_folder = find_test_folder(dataset_root)
    source_real_dir = test_folder / "real"
    source_fake_dir = test_folder / "fake"

    target_real_dir = jpg_root / "test" / "real"
    target_fake_dir = jpg_root / "test" / "fake"

    real_images = collect_images(source_real_dir)
    fake_images = collect_images(source_fake_dir)

    print("Original dataset root:", dataset_root)
    print("Detected test folder:", test_folder)
    print("Original real images found:", len(real_images))
    print("Original fake images found:", len(fake_images))

    if len(real_images) == 0:
        raise RuntimeError(f"No real images found in: {source_real_dir}")
    if len(fake_images) == 0:
        raise RuntimeError(f"No fake images found in: {source_fake_dir}")

    existing_real = len(list(target_real_dir.rglob("*.jpg"))) if target_real_dir.exists() else 0
    existing_fake = len(list(target_fake_dir.rglob("*.jpg"))) if target_fake_dir.exists() else 0

    if not force_rebuild and existing_real > 0 and existing_fake > 0:
        print("Using existing JPG-compatible folder:", jpg_root)
        print("Existing real JPG images:", existing_real)
        print("Existing fake JPG images:", existing_fake)
        return jpg_root

    target_real_dir.mkdir(parents=True, exist_ok=True)
    target_fake_dir.mkdir(parents=True, exist_ok=True)

    print("Creating JPG-compatible folder:", jpg_root)

    for i, src in enumerate(tqdm(real_images, desc="Converting real images")):
        out_path = target_real_dir / f"real_{i:06d}.jpg"
        try:
            img = Image.open(src).convert("RGB")
            img.save(out_path, quality=95)
        except Exception as e:
            print(f"[Warning] Skipped real image {src}: {e}")

    for i, src in enumerate(tqdm(fake_images, desc="Converting fake images")):
        out_path = target_fake_dir / f"fake_{i:06d}.jpg"
        try:
            img = Image.open(src).convert("RGB")
            img.save(out_path, quality=95)
        except Exception as e:
            print(f"[Warning] Skipped fake image {src}: {e}")

    final_real = len(list(target_real_dir.rglob("*.jpg")))
    final_fake = len(list(target_fake_dir.rglob("*.jpg")))

    print("JPG-compatible real images:", final_real)
    print("JPG-compatible fake images:", final_fake)

    if final_real == 0 or final_fake == 0:
        raise RuntimeError("JPG conversion failed. Real or fake compatible folder is empty.")

    return jpg_root


def evaluate_clean_only(
    model_path: Path,
    experiment_name: str,
    dataset_name: str,
    jpg_root: Path,
    result_dir: Path,
    device,
) -> pd.DataFrame:
    print("=" * 80)
    print(f"Evaluating: {experiment_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Model path: {model_path}")
    print(f"JPG root: {jpg_root}")
    print("=" * 80)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = dp.load_trained_model(str(model_path), device)
    criterion = dp.nn.BCEWithLogitsLoss()

    loader = dp.make_loader(
        frame_root=jpg_root,
        split="test",
        degradation="clean",
        severity="medium",
        shuffle=False,
    )

    metrics, pred_df = dp.evaluate_model(
        model,
        loader,
        criterion,
        device,
        threshold=dp.cfg.FAKE_THRESHOLD,
    )

    result_row = {
        "experiment": experiment_name,
        "dataset_tag": dataset_name,
        "degradation": "clean",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "loss": metrics["loss"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        "model_path": str(model_path),
        "test_root": str(jpg_root),
    }

    result_df = pd.DataFrame([result_row])

    result_path = result_dir / f"{experiment_name}_{dataset_name}_clean_result.csv"
    pred_path = result_dir / f"{experiment_name}_{dataset_name}_clean_predictions.csv"

    result_df.to_csv(result_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    print("Saved result CSV:", result_path)
    print("Saved prediction CSV:", pred_path)
    print(result_df)

    return result_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean-only cross-domain evaluation for baseline and consistency proposed model."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/D1/ait_users/staff122/deepfake_project/FaceForensics++_C23",
        help="Original FF++ C23 data root. Used only to keep pipeline config consistent.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project",
        help="Main output folder containing models and results.",
    )
    parser.add_argument(
        "--external_root",
        type=str,
        required=True,
        help="External image dataset root, e.g. /.../Celeb_V2",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Short dataset tag for output files, e.g. celeb_v2",
    )
    parser.add_argument(
        "--jpg_root",
        type=str,
        default="",
        help="Optional JPG-compatible output root. Default: output_root/<dataset_name>_jpg",
    )
    parser.add_argument(
        "--force_rebuild_jpg",
        action="store_true",
        help="Rebuild JPG-compatible dataset folder even if it already exists.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b4",
        choices=["efficientnet_b4", "efficientnet_b3"],
        help="Model architecture used for saved checkpoints.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=380,
        help="Image size used by the trained model. Use 380 for EfficientNet-B4.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="",
        help="Optional path to baseline model checkpoint.",
    )
    parser.add_argument(
        "--proposed_model",
        type=str,
        default="",
        help="Optional path to consistency proposed model checkpoint.",
    )
    parser.add_argument(
        "--include_old_da",
        action="store_true",
        help="Also evaluate the degradation-aware model da_efficientnet_b4_robust_best.pt.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    external_root = Path(args.external_root)
    dataset_name = args.dataset_name
    jpg_root = Path(args.jpg_root) if args.jpg_root else output_root / f"{dataset_name}_jpg"
    result_dir = output_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Configure imported pipeline
    dp.cfg.DATA_ROOT = str(data_root)
    dp.cfg.OUTPUT_ROOT = str(output_root)
    dp.cfg.DATA_FORMAT = "ffpp"
    dp.cfg.MODEL_NAME = args.model
    dp.cfg.IMG_SIZE = args.img_size
    dp.cfg.BATCH_SIZE = args.batch_size
    dp.cfg.NUM_WORKERS = args.num_workers

    print("DATA_ROOT:", dp.cfg.DATA_ROOT)
    print("OUTPUT_ROOT:", dp.cfg.OUTPUT_ROOT)
    print("EXTERNAL_ROOT:", external_root)
    print("DATASET_NAME:", dataset_name)
    print("JPG_ROOT:", jpg_root)
    print("MODEL:", dp.cfg.MODEL_NAME)
    print("IMG_SIZE:", dp.cfg.IMG_SIZE)
    print("BATCH_SIZE:", dp.cfg.BATCH_SIZE)

    jpg_root = convert_to_jpg_compatible_root(
        dataset_root=external_root,
        jpg_root=jpg_root,
        force_rebuild=args.force_rebuild_jpg,
    )

    device = dp.ensure_torch_imports()
    print("Device:", device)
    if dp.torch.cuda.is_available():
        print("GPU:", dp.torch.cuda.get_device_name(0))

    baseline_model = (
        Path(args.baseline_model)
        if args.baseline_model
        else output_root / "models" / f"baseline_{args.model}_clean_best.pt"
    )

    proposed_model = (
        Path(args.proposed_model)
        if args.proposed_model
        else output_root / "models" / f"consistency_{args.model}_robust_best.pt"
    )

    result_frames = []

    baseline_df = evaluate_clean_only(
        model_path=baseline_model,
        experiment_name=f"baseline_{args.model}_clean",
        dataset_name=dataset_name,
        jpg_root=jpg_root,
        result_dir=result_dir,
        device=device,
    )
    result_frames.append(baseline_df)

    if args.include_old_da:
        old_da_model = output_root / "models" / f"da_{args.model}_robust_best.pt"
        old_da_df = evaluate_clean_only(
            model_path=old_da_model,
            experiment_name=f"da_{args.model}_robust",
            dataset_name=dataset_name,
            jpg_root=jpg_root,
            result_dir=result_dir,
            device=device,
        )
        result_frames.append(old_da_df)

    proposed_df = evaluate_clean_only(
        model_path=proposed_model,
        experiment_name=f"consistency_{args.model}_robust",
        dataset_name=dataset_name,
        jpg_root=jpg_root,
        result_dir=result_dir,
        device=device,
    )
    result_frames.append(proposed_df)

    comparison_df = pd.concat(result_frames, ignore_index=True)
    comparison_path = result_dir / f"{dataset_name}_clean_cross_domain_comparison_consistency.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("=" * 80)
    print(f"{dataset_name} clean-only cross-domain evaluation completed.")
    print("Saved comparison CSV:", comparison_path)
    print("=" * 80)
    print(comparison_df[[
        "experiment",
        "dataset_tag",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "loss",
        "confusion_matrix",
    ]])


if __name__ == "__main__":
    main()
