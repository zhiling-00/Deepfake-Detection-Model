# Dataset DFDC: https://www.kaggle.com/competitions/deepfake-detection-challenge/data

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}


def sample_frame_indices(total_frames: int, n_samples: int):
    if total_frames <= 0:
        return []
    if total_frames <= n_samples:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, n_samples, dtype=int).tolist()


class FaceCropper:

    def __init__(self, face_detector: str = "haar", margin: float = 0.25):
        self.face_detector = face_detector.lower()
        self.margin = margin
        self.haar = None

        if self.face_detector == "haar":
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.haar.empty():
                print("[Warning] Haar detector failed to load. Falling back to center crop.")
                self.face_detector = "none"
        elif self.face_detector == "none":
            pass
        else:
            raise ValueError("--face_detector must be either 'haar' or 'none'.")

    @staticmethod
    def center_crop_square(frame):
        h, w = frame.shape[:2]
        side = min(h, w)
        y0 = max((h - side) // 2, 0)
        x0 = max((w - side) // 2, 0)
        return frame[y0:y0 + side, x0:x0 + side]

    def crop_face(self, frame):
        if self.face_detector == "none":
            return self.center_crop_square(frame)

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        if len(faces) == 0:
            return self.center_crop_square(frame)

        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])

        mx = int(fw * self.margin)
        my = int(fh * self.margin)

        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + fw + mx)
        y2 = min(h, y + fh + my)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self.center_crop_square(frame)

        return crop


def extract_video_frames(video_path: Path, out_dir: Path, cropper: FaceCropper,
                         frames_per_video: int, img_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Warning] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(total_frames, frames_per_video)

    saved_paths = []

    for i, frame_idx in enumerate(frame_indices):
        out_path = out_dir / f"frame_{i:04d}.jpg"

        if out_path.exists():
            saved_paths.append(out_path)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        face = cropper.crop_face(frame)
        face = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(out_path), face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


def load_metadata(source_dir: Path):
    metadata_path = source_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


def collect_available_videos(source_dir: Path, metadata: dict):
    real_videos = []
    fake_videos = []
    missing_videos = []

    for video_name, info in metadata.items():
        video_path = source_dir / video_name

        if not video_path.exists():
            missing_videos.append(video_name)
            continue

        if video_path.suffix.lower() not in VIDEO_EXTS:
            continue

        label = str(info.get("label", "")).upper()

        if label == "REAL":
            real_videos.append(video_name)
        elif label == "FAKE":
            fake_videos.append(video_name)

    return sorted(real_videos), sorted(fake_videos), missing_videos


def select_balanced_subset(real_videos, fake_videos, n_real: int, n_fake: int, seed: int):
    if len(real_videos) < n_real:
        raise RuntimeError(f"Not enough REAL videos. Need {n_real}, found {len(real_videos)}")

    if len(fake_videos) < n_fake:
        raise RuntimeError(f"Not enough FAKE videos. Need {n_fake}, found {len(fake_videos)}")

    rng = random.Random(seed)

    selected_real = sorted(real_videos)[:n_real]
    selected_fake = rng.sample(fake_videos, n_fake)

    return selected_real, selected_fake


def process_selected_videos(selected_videos, label_name: str, source_dir: Path,
                            output_root: Path, metadata: dict, cropper: FaceCropper,
                            frames_per_video: int, img_size: int):
    records = []

    for video_name in tqdm(selected_videos, desc=f"Extracting {label_name} videos"):
        video_path = source_dir / video_name
        out_dir = output_root / "test" / label_name.lower() / Path(video_name).stem

        saved_paths = extract_video_frames(
            video_path=video_path,
            out_dir=out_dir,
            cropper=cropper,
            frames_per_video=frames_per_video,
            img_size=img_size,
        )

        for frame_path in saved_paths:
            records.append({
                "video_name": video_name,
                "label": label_name.upper(),
                "frame_path": str(frame_path),
                "source_video": str(video_path),
                "original": metadata[video_name].get("original"),
            })

    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create balanced DFDC subset and extract face frames for cross-domain testing."
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        default="/media/D1/ait_users/staff122/deepfake_project/DFDC_chunk_0",
        help="Folder containing DFDC videos and metadata.json.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/media/D1/ait_users/staff122/deepfake_project/outputs_deepfake_project/DFDC_subset_77_frames",
        help="Output folder for extracted frames.",
    )
    parser.add_argument("--n_real", type=int, default=77, help="Number of REAL videos to select.")
    parser.add_argument("--n_fake", type=int, default=77, help="Number of FAKE videos to select.")
    parser.add_argument("--frames_per_video", type=int, default=32, help="Number of frames to extract per video.")
    parser.add_argument("--img_size", type=int, default=380, help="Output face frame size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fake video sampling.")
    parser.add_argument("--face_detector", type=str, default="haar", choices=["haar", "none"])
    parser.add_argument("--face_margin", type=float, default=0.25)

    return parser.parse_args()


def main():
    args = parse_args()

    source_dir = Path(args.source_dir)
    output_root = Path(args.output_root)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    metadata = load_metadata(source_dir)
    real_videos, fake_videos, missing_videos = collect_available_videos(source_dir, metadata)

    print("Source folder:", source_dir)
    print("Output folder:", output_root)
    print("Available REAL videos:", len(real_videos))
    print("Available FAKE videos:", len(fake_videos))
    print("Missing videos listed in metadata:", len(missing_videos))

    selected_real, selected_fake = select_balanced_subset(
        real_videos=real_videos,
        fake_videos=fake_videos,
        n_real=args.n_real,
        n_fake=args.n_fake,
        seed=args.seed,
    )

    cropper = FaceCropper(face_detector=args.face_detector, margin=args.face_margin)

    records = []
    records.extend(process_selected_videos(
        selected_videos=selected_real,
        label_name="REAL",
        source_dir=source_dir,
        output_root=output_root,
        metadata=metadata,
        cropper=cropper,
        frames_per_video=args.frames_per_video,
        img_size=args.img_size,
    ))

    records.extend(process_selected_videos(
        selected_videos=selected_fake,
        label_name="FAKE",
        source_dir=source_dir,
        output_root=output_root,
        metadata=metadata,
        cropper=cropper,
        frames_per_video=args.frames_per_video,
        img_size=args.img_size,
    ))

    output_root.mkdir(parents=True, exist_ok=True)

    metadata_csv = output_root / "dfdc_subset_frame_metadata.csv"
    selected_csv = output_root / "dfdc_subset_selected_videos.csv"

    frame_df = pd.DataFrame(records)
    frame_df.to_csv(metadata_csv, index=False)

    selected_df = pd.DataFrame(
        [{"video_name": v, "label": "REAL"} for v in selected_real]
        + [{"video_name": v, "label": "FAKE"} for v in selected_fake]
    )
    selected_df.to_csv(selected_csv, index=False)

    real_frame_count = len(list((output_root / "test" / "real").rglob("*.jpg")))
    fake_frame_count = len(list((output_root / "test" / "fake").rglob("*.jpg")))
    real_video_folder_count = len([p for p in (output_root / "test" / "real").glob("*") if p.is_dir()])
    fake_video_folder_count = len([p for p in (output_root / "test" / "fake").glob("*") if p.is_dir()])

    print("Done.")
    print("Selected REAL videos:", len(selected_real))
    print("Selected FAKE videos:", len(selected_fake))
    print("Real video folders:", real_video_folder_count)
    print("Fake video folders:", fake_video_folder_count)
    print("Real frames:", real_frame_count)
    print("Fake frames:", fake_frame_count)
    print("Frame metadata CSV:", metadata_csv)
    print("Selected videos CSV:", selected_csv)


if __name__ == "__main__":
    main()
