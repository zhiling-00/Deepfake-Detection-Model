import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import deepfake_detection_efficientnet as dp


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: Path):
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def choose_samples(frame_root: Path, num_per_class: int, seed: int = 42):
    rng = random.Random(seed)

    real_images = collect_images(frame_root / "test" / "real")
    fake_images = collect_images(frame_root / "test" / "fake")

    if len(real_images) == 0:
        raise RuntimeError(f"No real test images found in {frame_root / 'test' / 'real'}")
    if len(fake_images) == 0:
        raise RuntimeError(f"No fake test images found in {frame_root / 'test' / 'fake'}")

    real_selected = rng.sample(real_images, min(num_per_class, len(real_images)))
    fake_selected = rng.sample(fake_images, min(num_per_class, len(fake_images)))

    samples = []
    for p in real_selected:
        samples.append((p, 0, "real"))
    for p in fake_selected:
        samples.append((p, 1, "fake"))

    return samples


def get_target_layer(model):
    #Use the last EfficientNet feature block as Grad-CAM target.
    return model.features[-1]


def preprocess_image(img_path: Path, device):
    img = Image.open(img_path).convert("RGB")
    original = np.array(img)

    transform = dp.get_transforms(train=False)
    tensor = transform(img).unsqueeze(0).to(device)

    return tensor, original


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        prob = torch.sigmoid(logits)[0, 0]

        # Binary detector
        score = logits[0, 0]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

        gradients = self.gradients[0]       
        activations = self.activations[0]  

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, float(prob.detach().cpu().item())


def overlay_cam(original_rgb, cam, alpha=0.45):
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8((1 - alpha) * original_rgb + alpha * heatmap)
    return overlay, heatmap


def save_gradcam_figure(original_rgb, heatmap, overlay, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_gradcam_for_model(model_path: Path, model_name: str, samples, output_root: Path, device):
    print("=" * 80)
    print(f"Generating Grad-CAM for {model_name}")
    print("Model:", model_path)
    print("=" * 80)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = dp.load_trained_model(str(model_path), device)
    model.eval()

    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    out_dir = output_root / "gradcam" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, (img_path, label, label_name) in enumerate(samples):
        input_tensor, original_rgb = preprocess_image(img_path, device)
        cam, fake_prob = gradcam.generate(input_tensor)

        overlay, heatmap = overlay_cam(original_rgb, cam)

        pred_label = "fake" if fake_prob >= dp.cfg.FAKE_THRESHOLD else "real"
        correctness = "correct" if ((fake_prob >= dp.cfg.FAKE_THRESHOLD) == bool(label)) else "wrong"

        safe_name = f"{idx:02d}_{label_name}_pred_{pred_label}_prob_{fake_prob:.3f}_{correctness}.png"
        out_path = out_dir / safe_name

        title = (
            f"{model_name} | true={label_name} | pred={pred_label} | "
            f"fake_prob={fake_prob:.3f} | {correctness}"
        )

        save_gradcam_figure(original_rgb, heatmap, overlay, out_path, title)

        rows.append({
            "model": model_name,
            "image_path": str(img_path),
            "true_label": label,
            "true_label_name": label_name,
            "fake_probability": fake_prob,
            "pred_label": pred_label,
            "correctness": correctness,
            "gradcam_path": str(out_path),
        })

        print("Saved:", out_path)

    gradcam.remove_hooks()

    import pandas as pd
    csv_path = out_dir / f"{model_name}_gradcam_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Saved summary:", csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for baseline, DA, and consistency EfficientNet-B4 models.")

    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/D1/ait_users/staff122/deepfake_project/FaceForensics++_C23",
        help="FF++ data root. Used to configure the pipeline.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project",
        help="Output root containing preprocessed_image, models, and gradcam folders.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b4",
        choices=["efficientnet_b4", "efficientnet_b3"],
        help="Model architecture.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=380,
        help="Image size used during model training.",
    )
    parser.add_argument(
        "--num_per_class",
        type=int,
        default=5,
        help="Number of real and fake test images to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="",
        help="Optional baseline model checkpoint path.",
    )
    parser.add_argument(
        "--da_model",
        type=str,
        default="",
        help="Optional DA/degradation-aware model checkpoint path.",
    )
    parser.add_argument(
        "--consistency_model",
        type=str,
        default="",
        help="Optional consistency model checkpoint path.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    dataset_tag = data_root.name
    frame_root = output_root / "preprocessed_image" / dataset_tag

    dp.cfg.DATA_ROOT = str(data_root)
    dp.cfg.OUTPUT_ROOT = str(output_root)
    dp.cfg.DATA_FORMAT = "ffpp"
    dp.cfg.MODEL_NAME = args.model
    dp.cfg.IMG_SIZE = args.img_size

    device = dp.ensure_torch_imports()
    print("Device:", device)
    if dp.torch.cuda.is_available():
        print("GPU:", dp.torch.cuda.get_device_name(0))

    print("Frame root:", frame_root)

    samples = choose_samples(frame_root, args.num_per_class, args.seed)
    print(f"Selected {len(samples)} images for Grad-CAM.")

    baseline_model = (
        Path(args.baseline_model)
        if args.baseline_model
        else output_root / "models" / f"baseline_{args.model}_clean_best.pt"
    )

    da_model = (
        Path(args.da_model)
        if args.da_model
        else output_root / "models" / f"da_{args.model}_robust_best.pt"
    )

    consistency_model = (
        Path(args.consistency_model)
        if args.consistency_model
        else output_root / "models" / f"consistency_{args.model}_robust_best.pt"
    )

    models_to_run = [
        (baseline_model, f"baseline_{args.model}_clean"),
        (da_model, f"da_{args.model}_robust"),
        (consistency_model, f"consistency_{args.model}_robust"),
    ]

    for model_path, model_name in models_to_run:
        run_gradcam_for_model(
            model_path=model_path,
            model_name=model_name,
            samples=samples,
            output_root=output_root,
            device=device,
        )

    print("Grad-CAM generation completed for baseline, DA, and consistency models.")


if __name__ == "__main__":
    main()
