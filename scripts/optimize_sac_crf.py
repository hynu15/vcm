import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms
from tqdm import tqdm

from train_segmentation import load_segmentation_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
MODEL_PATH = PROJECT_ROOT / "models" / "best_pidnet.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "optimization"


def parse_int_list(text):
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def run_ffmpeg(cmd, step_name):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed at {step_name}:\n{result.stderr}")


def list_input_frames(image_dir, max_frames):
    files = []
    for city in sorted(os.listdir(image_dir)):
        city_dir = image_dir / city
        if not city_dir.is_dir():
            continue
        for fname in sorted(os.listdir(city_dir)):
            if fname.endswith("_leftImg8bit.png"):
                files.append(city_dir / fname)
                if len(files) >= max_frames:
                    return files
    return files


def load_model(device):
    model_path = MODEL_PATH
    if not model_path.is_file():
        legacy = PROJECT_ROOT / "models" / "best_ccnet.pth"
        if not legacy.is_file():
            raise FileNotFoundError(f"Model not found: {model_path} or {legacy}")
        model_path = legacy

    model, model_name = load_segmentation_model(str(model_path), device=device, num_classes=4)
    print(f"Using segmentation model: {model_name} | {model_path}")
    return model


def prepare_common_frames(files, model, device, tmp_frame_dir):
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    originals = []
    roi_masks = []

    tmp_frame_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(tqdm(files, desc="Preparing masks and split frames")):
        orig = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig)
        h, w = orig_np.shape[:2]

        input_tensor = transform(orig).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        roi_mask = (mask == 0).astype(np.uint8) * 255
        non_mask = 255 - roi_mask

        roi_img = cv2.bitwise_and(orig_np, orig_np, mask=roi_mask)
        non_img = cv2.bitwise_and(orig_np, orig_np, mask=non_mask)

        cv2.imwrite(str(tmp_frame_dir / f"frame_{idx:04d}_orig.png"), cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tmp_frame_dir / f"frame_{idx:04d}_roi.png"), cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tmp_frame_dir / f"frame_{idx:04d}_non.png"), cv2.cvtColor(non_img, cv2.COLOR_RGB2BGR))

        originals.append(orig_np)
        roi_masks.append((roi_mask > 0).astype(np.uint8))

    return originals, roi_masks


def file_bitrate_mbps(path, duration_sec):
    if duration_sec <= 0:
        return 0.0
    size_bits = path.stat().st_size * 8.0
    return (size_bits / duration_sec) / 1_000_000.0


def region_metrics(img1, img2, region_mask):
    if region_mask.sum() == 0:
        return 0.0, 0.0
    mask_3ch = np.repeat(region_mask[..., None], 3, axis=2)
    part1 = img1 * mask_3ch
    part2 = img2 * mask_3ch
    psnr = compare_psnr(part1, part2, data_range=255)
    ssim = compare_ssim(part1, part2, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def evaluate_combo(originals, roi_masks, sac_video, trad_video, crf_roi, crf_non):
    cap_sac = cv2.VideoCapture(str(sac_video))
    cap_trad = cv2.VideoCapture(str(trad_video))

    rows = []
    r_roi = crf_roi / (crf_roi + crf_non)
    r_non = crf_non / (crf_roi + crf_non)

    for idx, orig in enumerate(originals):
        ret_sac, sac_frame = cap_sac.read()
        ret_trad, trad_frame = cap_trad.read()
        if not ret_sac or not ret_trad:
            break

        sac_frame = cv2.cvtColor(sac_frame, cv2.COLOR_BGR2RGB)
        trad_frame = cv2.cvtColor(trad_frame, cv2.COLOR_BGR2RGB)

        psnr_trad = compare_psnr(orig, trad_frame, data_range=255)
        ssim_trad = compare_ssim(orig, trad_frame, channel_axis=2, data_range=255)
        psnr_sac = compare_psnr(orig, sac_frame, data_range=255)
        ssim_sac = compare_ssim(orig, sac_frame, channel_axis=2, data_range=255)

        roi_mask = roi_masks[idx]
        non_mask = 1 - roi_mask
        psnr_roi_sac, ssim_roi_sac = region_metrics(orig, sac_frame, roi_mask)
        psnr_non_sac, ssim_non_sac = region_metrics(orig, sac_frame, non_mask)

        sa_psnr = r_non * psnr_roi_sac + r_roi * psnr_non_sac
        sa_ssim = r_non * ssim_roi_sac + r_roi * ssim_non_sac

        rows.append(
            {
                "Frame": idx + 1,
                "PSNR_Trad": psnr_trad,
                "SSIM_Trad": ssim_trad,
                "PSNR_SAC": psnr_sac,
                "SSIM_SAC": ssim_sac,
                "SA-PSNR": sa_psnr,
                "SA-SSIM": sa_ssim,
            }
        )

    cap_sac.release()
    cap_trad.release()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No metrics computed. Check encoded videos and input frames.")
    return df


def encode_combo(tmp_frame_dir, combo_dir, crf_roi, crf_non, fps, preset):
    combo_dir.mkdir(parents=True, exist_ok=True)
    roi_video = combo_dir / "roi.mp4"
    non_video = combo_dir / "nonroi.mp4"
    sac_video = combo_dir / "sac_x265.mp4"
    trad_video = combo_dir / "traditional_x265.mp4"

    total_crf = int(round((crf_roi + crf_non) / 2.0))

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_frame_dir / "frame_%04d_roi.png"),
            "-c:v",
            "libx265",
            "-crf",
            str(crf_roi),
            "-preset",
            preset,
            str(roi_video),
        ],
        f"Encode ROI CRF={crf_roi}",
    )

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_frame_dir / "frame_%04d_non.png"),
            "-c:v",
            "libx265",
            "-crf",
            str(crf_non),
            "-preset",
            preset,
            str(non_video),
        ],
        f"Encode non-ROI CRF={crf_non}",
    )

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(roi_video),
            "-i",
            str(non_video),
            "-filter_complex",
            "[0:v][1:v]blend=all_mode=addition",
            str(sac_video),
        ],
        "Merge SAC video",
    )

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_frame_dir / "frame_%04d_orig.png"),
            "-c:v",
            "libx265",
            "-crf",
            str(total_crf),
            "-preset",
            preset,
            str(trad_video),
        ],
        f"Encode traditional CRF={total_crf}",
    )

    return roi_video, non_video, sac_video, trad_video


def main():
    parser = argparse.ArgumentParser(description="Grid-search CRF pairs for SAC-X265 optimization")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--roi-crf-list", type=str, default="20,23,25")
    parser.add_argument("--non-crf-list", type=str, default="30,32,35")
    parser.add_argument("--objective-lambda", type=float, default=0.4)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {DATA_DIR}")

    roi_crf_list = parse_int_list(args.roi_crf_list)
    non_crf_list = parse_int_list(args.non_crf_list)
    if not roi_crf_list or not non_crf_list:
        raise ValueError("CRF lists must not be empty")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_tag
    tmp_frame_dir = run_dir / "tmp_frames"
    combos_dir = run_dir / "combos"
    metrics_dir = run_dir / "metrics"

    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    files = list_input_frames(DATA_DIR, args.num_frames)
    if not files:
        raise RuntimeError("No input frames found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    originals, roi_masks = prepare_common_frames(files, model, device, tmp_frame_dir)

    duration_sec = len(originals) / float(args.fps)
    rows = []

    combos = [(a, b) for a in roi_crf_list for b in non_crf_list]

    for crf_roi, crf_non in tqdm(combos, desc="Evaluating CRF combos"):
        combo_name = f"roi{crf_roi}_non{crf_non}"
        combo_dir = combos_dir / combo_name
        roi_video, non_video, sac_video, trad_video = encode_combo(
            tmp_frame_dir=tmp_frame_dir,
            combo_dir=combo_dir,
            crf_roi=crf_roi,
            crf_non=crf_non,
            fps=args.fps,
            preset=args.preset,
        )

        df = evaluate_combo(
            originals=originals,
            roi_masks=roi_masks,
            sac_video=sac_video,
            trad_video=trad_video,
            crf_roi=crf_roi,
            crf_non=crf_non,
        )

        mean_metrics = df.mean(numeric_only=True)

        sac_bitrate = file_bitrate_mbps(roi_video, duration_sec) + file_bitrate_mbps(non_video, duration_sec)
        trad_bitrate = file_bitrate_mbps(trad_video, duration_sec)
        delta_sa_psnr = float(mean_metrics["SA-PSNR"] - mean_metrics["PSNR_Trad"])
        delta_sa_ssim = float(mean_metrics["SA-SSIM"] - mean_metrics["SSIM_Trad"])
        bitrate_delta = float(sac_bitrate - trad_bitrate)
        score = delta_sa_psnr - args.objective_lambda * bitrate_delta

        combo_metrics_path = metrics_dir / f"metrics_{combo_name}.csv"
        df.to_csv(combo_metrics_path, index=False)

        rows.append(
            {
                "crf_roi": crf_roi,
                "crf_non": crf_non,
                "crf_trad": int(round((crf_roi + crf_non) / 2.0)),
                "PSNR_Trad": float(mean_metrics["PSNR_Trad"]),
                "SSIM_Trad": float(mean_metrics["SSIM_Trad"]),
                "PSNR_SAC": float(mean_metrics["PSNR_SAC"]),
                "SSIM_SAC": float(mean_metrics["SSIM_SAC"]),
                "SA-PSNR": float(mean_metrics["SA-PSNR"]),
                "SA-SSIM": float(mean_metrics["SA-SSIM"]),
                "delta_SA_PSNR_vs_Trad": delta_sa_psnr,
                "delta_SA_SSIM_vs_Trad": delta_sa_ssim,
                "sac_bitrate_mbps": sac_bitrate,
                "trad_bitrate_mbps": trad_bitrate,
                "bitrate_delta_mbps": bitrate_delta,
                "objective_score": score,
                "per_frame_metrics_csv": str(combo_metrics_path.relative_to(PROJECT_ROOT)),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(by="objective_score", ascending=False)
    summary_path = metrics_dir / "crf_optimization_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best = summary_df.iloc[0]
    baseline_mask = (summary_df["crf_roi"] == 23) & (summary_df["crf_non"] == 32)
    baseline = summary_df[baseline_mask].iloc[0] if baseline_mask.any() else None

    report_lines = [
        "SAC CRF Optimization Report",
        f"run_dir: {run_dir.relative_to(PROJECT_ROOT)}",
        f"num_frames: {len(originals)}",
        f"objective: delta_SA_PSNR_vs_Trad - lambda * bitrate_delta_mbps",
        f"lambda: {args.objective_lambda}",
        "",
        "Best configuration:",
        f"- crf_roi={int(best['crf_roi'])}, crf_non={int(best['crf_non'])}, crf_trad={int(best['crf_trad'])}",
        f"- SA-PSNR={best['SA-PSNR']:.3f}, PSNR_Trad={best['PSNR_Trad']:.3f}, delta={best['delta_SA_PSNR_vs_Trad']:+.3f}",
        f"- SA-SSIM={best['SA-SSIM']:.4f}, SSIM_Trad={best['SSIM_Trad']:.4f}, delta={best['delta_SA_SSIM_vs_Trad']:+.4f}",
        f"- SAC bitrate={best['sac_bitrate_mbps']:.3f} Mbps, Trad bitrate={best['trad_bitrate_mbps']:.3f} Mbps, delta={best['bitrate_delta_mbps']:+.3f} Mbps",
        f"- objective_score={best['objective_score']:.3f}",
        "",
    ]

    if baseline is not None:
        report_lines.extend(
            [
                "Baseline comparison (23/32):",
                f"- baseline objective_score={baseline['objective_score']:.3f}",
                f"- gain over baseline={best['objective_score'] - baseline['objective_score']:+.3f}",
                f"- baseline delta SA-PSNR={baseline['delta_SA_PSNR_vs_Trad']:+.3f}",
            ]
        )

    report_path = metrics_dir / "best_config_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n[RESULT] Top 5 configurations by objective score")
    print(summary_df.head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[RESULT] Summary CSV: {summary_path}")
    print(f"[RESULT] Text report: {report_path}")

    if not args.keep_artifacts:
        shutil.rmtree(tmp_frame_dir, ignore_errors=True)
        print("[INFO] Removed temporary frame artifacts")


if __name__ == "__main__":
    main()
