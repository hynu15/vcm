import argparse
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from train_segmentation import Cityscapes4Class, build_segmentation_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "gt_4class" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
VAL_LABEL_DIR = PROJECT_ROOT / "data" / "gt_4class" / "val"


@dataclass
class BenchmarkResult:
	training_csv: Path
	training_summary_json: Path
	metrics_per_frame_csv: Path
	metrics_summary_csv: Path
	latency_csv: Path
	checkpoint_path: Path


def run_ffmpeg(cmd: List[str], step_name: str) -> None:
	result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	if result.returncode != 0:
		raise RuntimeError(f"FFmpeg failed at {step_name}:\n{result.stderr}")


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
	bce = nn.CrossEntropyLoss()(pred, target)
	pred_soft = F.softmax(pred, dim=1)
	target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
	dice = 1 - (
		(2 * (pred_soft * target_onehot).sum(dim=(2, 3)) + 1)
		/ (pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3)) + 1)
	)
	return bce + dice.mean()


def _build_dataset(image_size: Tuple[int, int]) -> Cityscapes4Class:
	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	return Cityscapes4Class(
		str(VAL_IMAGE_DIR),
		str(VAL_LABEL_DIR),
		transform=transform,
		image_size=image_size,
	)


def _split_dataset(dataset: Cityscapes4Class, train_ratio: float, max_samples: int, seed: int):
	total = len(dataset)
	indices = list(range(total))
	rng = random.Random(seed)
	rng.shuffle(indices)
	if max_samples > 0:
		indices = indices[: min(max_samples, len(indices))]
	train_size = max(1, int(len(indices) * train_ratio))
	train_idx = indices[:train_size]
	val_idx = indices[train_size:]
	if not val_idx:
		val_idx = train_idx[-1:]
		train_idx = train_idx[:-1]
	return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_and_log(
	model_name: str,
	run_dir: Path,
	device: torch.device,
	epochs: int,
	batch_size: int,
	num_workers: int,
	lr: float,
	weight_decay: float,
	image_size: Tuple[int, int],
	train_ratio: float,
	max_samples: int,
	seed: int,
) -> Tuple[Path, Path, Path]:
	set_seed(seed)
	dataset = _build_dataset(image_size)
	train_ds, val_ds = _split_dataset(dataset, train_ratio, max_samples, seed)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=(device.type == "cuda"),
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=(device.type == "cuda"),
	)

	model = build_segmentation_model(model_name=model_name, num_classes=4, backbone_weights=None).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

	run_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = run_dir / f"best_{model_name}.pth"
	csv_path = run_dir / f"training_{model_name}.csv"
	summary_path = run_dir / f"training_{model_name}_summary.json"

	rows = []
	best_miou = -1.0
	best_epoch = 0

	for epoch in range(1, epochs + 1):
		model.train()
		train_losses = []
		for images, labels in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs}", leave=False):
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)
			with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
				outputs = model(images)
				loss = bce_dice_loss(outputs, labels)

			if not torch.isfinite(loss):
				continue

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			train_losses.append(float(loss.item()))

		model.eval()
		ious = []
		with torch.no_grad():
			for images, labels in val_loader:
				images = images.to(device, non_blocking=True)
				labels = labels.to(device, non_blocking=True)
				logits = model(images)
				pred = torch.argmax(logits, dim=1)
				for p, t in zip(pred, labels):
					per_class_iou = []
					for c in range(4):
						inter = ((p == c) & (t == c)).sum().item()
						union = ((p == c) | (t == c)).sum().item()
						per_class_iou.append(inter / union if union > 0 else 0.0)
					ious.append(float(np.mean(per_class_iou)))

		finite_losses = [value for value in train_losses if np.isfinite(value)]
		train_loss = float(np.mean(finite_losses)) if finite_losses else 0.0
		val_miou = float(np.mean(ious)) if ious else 0.0
		rows.append({"epoch": epoch, "train_loss": train_loss, "val_miou": val_miou})

		if val_miou > best_miou:
			best_miou = val_miou
			best_epoch = epoch
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"meta": {"model_name": model_name, "num_classes": 4},
					"epoch": epoch,
					"best_miou": best_miou,
				},
				checkpoint_path,
			)

	df = pd.DataFrame(rows)
	df.to_csv(csv_path, index=False)
	summary = {
		"model_name": model_name,
		"device": str(device),
		"seed": seed,
		"samples_total": len(train_ds) + len(val_ds),
		"samples_train": len(train_ds),
		"samples_val": len(val_ds),
		"epochs": epochs,
		"best_epoch": best_epoch,
		"best_val_miou": best_miou,
		"checkpoint": str(checkpoint_path),
	}
	pd.Series(summary).to_json(summary_path, indent=2)
	return csv_path, summary_path, checkpoint_path


def _list_eval_frames(num_frames: int) -> List[Path]:
	paths: List[Path] = []
	for city in sorted(os.listdir(VAL_IMAGE_DIR)):
		city_dir = VAL_IMAGE_DIR / city
		if not city_dir.is_dir():
			continue
		for fname in sorted(os.listdir(city_dir)):
			if fname.endswith("_leftImg8bit.png"):
				paths.append(city_dir / fname)
				if len(paths) >= num_frames:
					return paths
	return paths


def _prepare_model_for_infer(model_name: str, checkpoint_path: Path, device: torch.device):
	model = build_segmentation_model(model_name=model_name, num_classes=4, backbone_weights=None).to(device)
	ckpt = torch.load(checkpoint_path, map_location=device)
	state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
	model.load_state_dict(state_dict)
	model.eval()
	return model


def run_sac_metrics(
	model_name: str,
	checkpoint_path: Path,
	run_dir: Path,
	device: torch.device,
	num_frames: int,
	crf_roi: int,
	crf_non: int,
	preset: str,
) -> Tuple[Path, Path]:
	transform = transforms.Compose(
		[
			transforms.Resize((512, 1024)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)

	model = _prepare_model_for_infer(model_name, checkpoint_path, device)
	frames = _list_eval_frames(num_frames)
	if not frames:
		raise RuntimeError("No evaluation frames found")

	tmp_dir = run_dir / "tmp_metrics_frames"
	tmp_dir.mkdir(parents=True, exist_ok=True)
	fps = 30

	originals = []
	roi_masks = []
	for idx, path in enumerate(frames):
		orig = Image.open(path).convert("RGB")
		orig_np = np.array(orig)
		h, w = orig_np.shape[:2]
		originals.append(orig_np)

		with torch.no_grad():
			inp = transform(orig).unsqueeze(0).to(device)
			pred = model(inp)
		mask = torch.argmax(pred, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
		mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
		roi = (mask == 0).astype(np.uint8) * 255
		roi_masks.append((roi > 0).astype(np.uint8))

		roi_img = cv2.bitwise_and(orig_np, orig_np, mask=roi)
		non_img = cv2.bitwise_and(orig_np, orig_np, mask=255 - roi)
		cv2.imwrite(str(tmp_dir / f"frame_{idx:04d}_orig.png"), cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
		cv2.imwrite(str(tmp_dir / f"frame_{idx:04d}_roi.png"), cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
		cv2.imwrite(str(tmp_dir / f"frame_{idx:04d}_non.png"), cv2.cvtColor(non_img, cv2.COLOR_RGB2BGR))

	roi_video = run_dir / "roi.mp4"
	non_video = run_dir / "nonroi.mp4"
	sac_video = run_dir / "sac_x265.mp4"
	trad_video = run_dir / "traditional_x265.mp4"
	crf_trad = int(round((crf_roi + crf_non) / 2.0))

	run_ffmpeg(
		[
			"ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / "frame_%04d_roi.png"),
			"-c:v", "libx265", "-crf", str(crf_roi), "-preset", preset, str(roi_video),
		],
		"encode roi",
	)
	run_ffmpeg(
		[
			"ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / "frame_%04d_non.png"),
			"-c:v", "libx265", "-crf", str(crf_non), "-preset", preset, str(non_video),
		],
		"encode non",
	)
	run_ffmpeg(
		[
			"ffmpeg", "-y", "-i", str(roi_video), "-i", str(non_video),
			"-filter_complex", "[0:v][1:v]blend=all_mode=addition", str(sac_video),
		],
		"merge sac",
	)
	run_ffmpeg(
		[
			"ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / "frame_%04d_orig.png"),
			"-c:v", "libx265", "-crf", str(crf_trad), "-preset", preset, str(trad_video),
		],
		"encode trad",
	)

	cap_sac = cv2.VideoCapture(str(sac_video))
	cap_trad = cv2.VideoCapture(str(trad_video))
	rows = []
	r_roi = crf_roi / (crf_roi + crf_non)
	r_non = crf_non / (crf_roi + crf_non)

	for idx, orig in enumerate(originals):
		ok_sac, sac_frame = cap_sac.read()
		ok_trad, trad_frame = cap_trad.read()
		if not ok_sac or not ok_trad:
			break
		sac_frame = cv2.cvtColor(sac_frame, cv2.COLOR_BGR2RGB)
		trad_frame = cv2.cvtColor(trad_frame, cv2.COLOR_BGR2RGB)

		psnr_trad = compare_psnr(orig, trad_frame, data_range=255)
		ssim_trad = compare_ssim(orig, trad_frame, channel_axis=2, data_range=255)
		psnr_sac = compare_psnr(orig, sac_frame, data_range=255)
		ssim_sac = compare_ssim(orig, sac_frame, channel_axis=2, data_range=255)

		roi_mask = roi_masks[idx]
		non_mask = 1 - roi_mask
		roi3 = np.repeat(roi_mask[:, :, None], 3, axis=2)
		non3 = np.repeat(non_mask[:, :, None], 3, axis=2)
		psnr_roi = compare_psnr(orig * roi3, sac_frame * roi3, data_range=255)
		ssim_roi = compare_ssim(orig * roi3, sac_frame * roi3, channel_axis=2, data_range=255)
		psnr_non = compare_psnr(orig * non3, sac_frame * non3, data_range=255)
		ssim_non = compare_ssim(orig * non3, sac_frame * non3, channel_axis=2, data_range=255)

		sa_psnr = r_non * psnr_roi + r_roi * psnr_non
		sa_ssim = r_non * ssim_roi + r_roi * ssim_non

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

	per_frame = pd.DataFrame(rows)
	per_frame_csv = run_dir / f"metrics_{model_name}_per_frame.csv"
	summary_csv = run_dir / f"metrics_{model_name}_summary.csv"
	per_frame.to_csv(per_frame_csv, index=False)

	duration_sec = len(originals) / float(fps)
	sac_bitrate = (roi_video.stat().st_size * 8 + non_video.stat().st_size * 8) / duration_sec / 1_000_000.0
	trad_bitrate = trad_video.stat().st_size * 8 / duration_sec / 1_000_000.0
	means = per_frame.mean(numeric_only=True)
	summary_row = {
		"model": model_name,
		"num_frames": len(per_frame),
		"crf_roi": crf_roi,
		"crf_non": crf_non,
		"crf_trad": crf_trad,
		"PSNR_Trad": float(means["PSNR_Trad"]),
		"SSIM_Trad": float(means["SSIM_Trad"]),
		"PSNR_SAC": float(means["PSNR_SAC"]),
		"SSIM_SAC": float(means["SSIM_SAC"]),
		"SA-PSNR": float(means["SA-PSNR"]),
		"SA-SSIM": float(means["SA-SSIM"]),
		"delta_SA_PSNR_vs_Trad": float(means["SA-PSNR"] - means["PSNR_Trad"]),
		"delta_SA_SSIM_vs_Trad": float(means["SA-SSIM"] - means["SSIM_Trad"]),
		"sac_bitrate_mbps": sac_bitrate,
		"trad_bitrate_mbps": trad_bitrate,
		"bitrate_delta_mbps": sac_bitrate - trad_bitrate,
	}
	pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

	shutil.rmtree(tmp_dir, ignore_errors=True)
	return per_frame_csv, summary_csv


def run_latency_benchmark(
	model_name: str,
	checkpoint_path: Path,
	run_dir: Path,
	num_frames: int,
	crf_roi: int,
	crf_non: int,
	preset: str,
	image_size: Tuple[int, int],
) -> Path:
	frames = _list_eval_frames(num_frames)
	if not frames:
		raise RuntimeError("No frames for latency benchmark")

	transform = transforms.Compose(
		[
			transforms.Resize(image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)

	model_gpu = None
	if torch.cuda.is_available():
		model_gpu = _prepare_model_for_infer(model_name, checkpoint_path, torch.device("cuda"))
	model_cpu = _prepare_model_for_infer(model_name, checkpoint_path, torch.device("cpu"))

	stage_names = ["preprocessing", "inference", "mask_split", "x265_encode_decode", "blend", "total"]
	accum: Dict[str, Dict[str, float]] = {
		s: {"gpu": 0.0, "cpu": 0.0} for s in stage_names
	}

	tmp_dir = run_dir / "tmp_latency"
	tmp_dir.mkdir(parents=True, exist_ok=True)
	fps = 30
	crf_trad = int(round((crf_roi + crf_non) / 2.0))

	for device_name, model, device in [
		("gpu", model_gpu, torch.device("cuda") if model_gpu is not None else None),
		("cpu", model_cpu, torch.device("cpu")),
	]:
		if model is None or device is None:
			continue

		for idx, frame_path in enumerate(frames):
			t_total = time.perf_counter()
			orig = Image.open(frame_path).convert("RGB")
			orig_np = np.array(orig)
			h, w = orig_np.shape[:2]

			t0 = time.perf_counter()
			inp = transform(orig).unsqueeze(0).to(device)
			if device_name == "gpu":
				torch.cuda.synchronize()
			accum["preprocessing"][device_name] += (time.perf_counter() - t0) * 1000.0

			t1 = time.perf_counter()
			with torch.no_grad():
				pred = model(inp)
			if device_name == "gpu":
				torch.cuda.synchronize()
			accum["inference"][device_name] += (time.perf_counter() - t1) * 1000.0

			t2 = time.perf_counter()
			mask = torch.argmax(pred, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
			mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
			roi = (mask == 0).astype(np.uint8) * 255
			roi_img = cv2.bitwise_and(orig_np, orig_np, mask=roi)
			non_img = cv2.bitwise_and(orig_np, orig_np, mask=255 - roi)
			accum["mask_split"][device_name] += (time.perf_counter() - t2) * 1000.0

			cv2.imwrite(str(tmp_dir / f"{device_name}_{idx:04d}_orig.png"), cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
			cv2.imwrite(str(tmp_dir / f"{device_name}_{idx:04d}_roi.png"), cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
			cv2.imwrite(str(tmp_dir / f"{device_name}_{idx:04d}_non.png"), cv2.cvtColor(non_img, cv2.COLOR_RGB2BGR))

			t3 = time.perf_counter()
			roi_video = tmp_dir / f"{device_name}_{idx:04d}_roi.mp4"
			non_video = tmp_dir / f"{device_name}_{idx:04d}_non.mp4"
			trad_video = tmp_dir / f"{device_name}_{idx:04d}_trad.mp4"
			run_ffmpeg(
				[
					"ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / f"{device_name}_{idx:04d}_roi.png"),
					"-c:v", "libx265", "-crf", str(crf_roi), "-preset", preset, str(roi_video),
				],
				f"latency roi {device_name} frame {idx}",
			)
			run_ffmpeg(
				[
					"ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / f"{device_name}_{idx:04d}_non.png"),
					"-c:v", "libx265", "-crf", str(crf_non), "-preset", preset, str(non_video),
				],
				f"latency non {device_name} frame {idx}",
			)
			run_ffmpeg(
				[
					"ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / f"{device_name}_{idx:04d}_orig.png"),
					"-c:v", "libx265", "-crf", str(crf_trad), "-preset", preset, str(trad_video),
				],
				f"latency trad {device_name} frame {idx}",
			)

			cap_roi = cv2.VideoCapture(str(roi_video))
			cap_non = cv2.VideoCapture(str(non_video))
			ok_r, r = cap_roi.read()
			ok_n, n = cap_non.read()
			cap_roi.release()
			cap_non.release()
			if not ok_r or not ok_n:
				raise RuntimeError("Unable to decode temporary latency videos")
			r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
			n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
			accum["x265_encode_decode"][device_name] += (time.perf_counter() - t3) * 1000.0

			t4 = time.perf_counter()
			roi3 = np.repeat((roi > 0).astype(np.uint8)[:, :, None], 3, axis=2)
			_merged = np.where(roi3 == 1, r, n)
			accum["blend"][device_name] += (time.perf_counter() - t4) * 1000.0
			accum["total"][device_name] += (time.perf_counter() - t_total) * 1000.0

	rows = []
	for stage in stage_names:
		row = {
			"step": stage,
			"gpu_ms": accum[stage]["gpu"] / max(1, num_frames) if model_gpu is not None else np.nan,
			"cpu_ms": accum[stage]["cpu"] / max(1, num_frames),
		}
		rows.append(row)

	df = pd.DataFrame(rows)
	total_gpu = df.loc[df["step"] == "total", "gpu_ms"].iloc[0]
	total_cpu = df.loc[df["step"] == "total", "cpu_ms"].iloc[0]
	fps_gpu = 1000.0 / total_gpu if model_gpu is not None and total_gpu > 0 else np.nan
	fps_cpu = 1000.0 / total_cpu if total_cpu > 0 else np.nan
	df = pd.concat(
		[
			df,
			pd.DataFrame(
				[
					{"step": "fps", "gpu_ms": fps_gpu, "cpu_ms": fps_cpu},
				]
			),
		],
		ignore_index=True,
	)

	latency_csv = run_dir / f"latency_{model_name}.csv"
	df.to_csv(latency_csv, index=False)
	shutil.rmtree(tmp_dir, ignore_errors=True)
	return latency_csv


def run_full_benchmark(args: argparse.Namespace) -> BenchmarkResult:
	model_dir_name = args.model_name.replace("/", "_")
	run_name = getattr(args, "run_name", None)
	if run_name:
		run_dir = PROJECT_ROOT / "outputs" / "model_benchmark" / run_name
	else:
		run_dir = PROJECT_ROOT / "outputs" / "model_benchmark" / model_dir_name
	run_dir.mkdir(parents=True, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	training_csv, training_summary_json, checkpoint_path = train_and_log(
		model_name=args.model_name,
		run_dir=run_dir,
		device=device,
		epochs=args.epochs,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		lr=args.lr,
		weight_decay=args.weight_decay,
		image_size=(args.train_height, args.train_width),
		train_ratio=args.train_ratio,
		max_samples=args.max_samples,
		seed=args.seed,
	)

	metrics_per_frame_csv, metrics_summary_csv = run_sac_metrics(
		model_name=args.model_name,
		checkpoint_path=checkpoint_path,
		run_dir=run_dir,
		device=device,
		num_frames=args.eval_frames,
		crf_roi=args.crf_roi,
		crf_non=args.crf_non,
		preset=args.preset,
	)

	latency_csv = run_latency_benchmark(
		model_name=args.model_name,
		checkpoint_path=checkpoint_path,
		run_dir=run_dir,
		num_frames=args.latency_frames,
		crf_roi=args.crf_roi,
		crf_non=args.crf_non,
		preset=args.latency_preset,
		image_size=(args.latency_height, args.latency_width),
	)

	return BenchmarkResult(
		training_csv=training_csv,
		training_summary_json=training_summary_json,
		metrics_per_frame_csv=metrics_per_frame_csv,
		metrics_summary_csv=metrics_summary_csv,
		latency_csv=latency_csv,
		checkpoint_path=checkpoint_path,
	)

