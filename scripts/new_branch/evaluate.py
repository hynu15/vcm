"""Reproduce Bảng II + Bảng IV của paper (Wang et al. 2023) — bản cải tiến.

Chạy 4 phương pháp:
- H264     CRF (23, 23)
- SA-X264  CRF (18, 27)
- H265     CRF (28, 28)
- SA-X265  CRF (23, 32)

Trên test split (strasbourg + ulm = 460 ảnh). Chia thành "GOP" mỗi 30 frame.

Cải tiến so với bản gốc (theo report/so_sanh_paper_vs_new_branch.tex):
1. `--roi-source {pred,gt}`  — Đề xuất #1: dùng GT mask thay vì pred để so fair với paper-folder
   (paper-folder TwoStream_generate.py đọc thẳng _gtFine_4class.png).
2. `--no-gray`                — Đề xuất #2: giữ RGB khi predict seg trên frame tái tạo
   (kênh màu quan trọng cho lớp sky/nature).
3. `--framerate INT`          — Đề xuất #7: mặc định 17 fps (đúng Cityscapes seq), không phải 30.
4. `--keyint INT`             — Đề xuất #7: 0 = để codec tự chọn GOP (mặc định x264 ~250). Trước
   đây hardcode 30 → mỗi giây 1 I-frame, tốn bitrate.
5. `--match-bitrate`          — Đề xuất #8: với SA-X264/SA-X265, binary search crf_non để
   khớp bytes với baseline H.264/H.265 (so sánh fair theo paper Bảng IV).
6. `miou_before / iiou_before`— Đề xuất #9: đo seg trên frame gốc → suy ra drop do nén
   (miou_drop = before − after). Đây mới là chỉ số đo trực tiếp tác dụng SAC.

Output (outputs/new_branch/eval/<timestamp>/):
- per_frame.csv – per-frame metric
- summary.csv   – tổng kết theo phương pháp (paper-style Bảng II/IV)
- summary.json
"""
from __future__ import annotations

# Cho phép chạy cả `python -m scripts.new_branch.evaluate` lẫn `python evaluate.py`
if __package__ in (None, ""):
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    __package__ = "scripts.new_branch"

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image

from . import metrics as M
from .compression import compress_sac, compress_traditional
from .dataset import CityscapesSAC, _normalize, _rgb_to_gray3
from .model import build_pidnet_l, build_segnet
from .paths import (CRF_CONFIGS, EVAL_DIR, FRAME_HW, NUM_CLASSES,
                    PIDNET_L_CHECKPOINT, ROI_CLASS_ID, SAC_CHECKPOINT, SEG_INPUT_HW)
from .stream_separation import build_roi_mask, gop_roi_ratio, select_delta_crf


@dataclass
class FrameRecord:
    method: str
    crf_roi: int
    crf_non: int
    idx: int
    name: str
    psnr_full: float
    ssim_full: float
    psnr_roi: float
    ssim_roi: float
    psnr_non: float
    ssim_non: float
    sa_psnr: float
    sa_ssim: float
    miou_before: float
    iiou_before: float
    miou_after: float
    iiou_after: float
    miou_drop: float
    iiou_drop: float
    bytes_total: int
    roi_ratio_gop: float = 0.0  # RA-CRF: tỷ lệ ROI trung bình của GOP chứa frame này
    delta_crf: int = 0          # RA-CRF: ΔCRF đã dùng (0 nếu không phải adaptive)


def _to_full_res(img_lowres: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Upsample uint8 RGB từ SEG_INPUT_HW lên full-res."""
    return np.array(Image.fromarray(img_lowres).resize((size_hw[1], size_hw[0]), Image.BILINEAR))


@torch.no_grad()
def _seg_predict(model, img_tensor_lowres: torch.Tensor, device) -> np.ndarray:
    """img_tensor: (3,H,W) normalized; trả về (H,W) int8 prediction."""
    img = img_tensor_lowres.unsqueeze(0).to(device)
    use_amp = device.type == "cuda"
    with autocast(device_type="cuda", enabled=use_amp):
        out = model(img)
        logits = out[0] if isinstance(out, (list, tuple)) else out
    H, W = SEG_INPUT_HW
    logits = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=True)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def _frame_to_seg_tensor(frame_full_rgb: np.ndarray, use_gray: bool) -> torch.Tensor:
    """RGB full-res uint8 → tensor sẵn sàng cho mạng seg.

    - Resize về SEG_INPUT_HW.
    - Nếu use_gray: convert grayscale-as-3ch (paper CLAUDE.md §5.1).
    - Normalize ImageNet.
    """
    H, W = SEG_INPUT_HW
    lowres = np.array(Image.fromarray(frame_full_rgb).resize((W, H), Image.BILINEAR))
    if use_gray:
        lowres = _rgb_to_gray3(lowres)
    return _normalize(lowres)


def _gop_chunks(items: list, gop: int):
    for i in range(0, len(items), gop):
        yield items[i:i + gop]


def _binary_search_crf(
    frames_sample: list[np.ndarray],
    masks_sample: list[np.ndarray],
    codec: str,
    crf_roi: int,
    target_bytes: int,
    framerate: int,
    keyint: int | None,
    tol: float = 0.02,
    max_iter: int = 8,
) -> int:
    """Tìm crf_non ∈ [crf_roi+1, 51] sao cho SAC bytes ≈ target_bytes ±tol.

    Dùng sample (1 GOP) để tránh chi phí. Trả về crf_non tối ưu (int).
    """
    lo, hi = crf_roi + 1, 51
    best_crf, best_diff = hi, float("inf")
    for it in range(max_iter):
        mid = (lo + hi) // 2
        _, b_r, b_n = compress_sac(frames_sample, masks_sample, codec,
                                   crf_roi=crf_roi, crf_non=mid,
                                   framerate=framerate, keyint=keyint)
        total = b_r + b_n
        diff = abs(total - target_bytes) / max(target_bytes, 1)
        print(f"    [match-bitrate iter {it+1}] crf_non={mid:2d}  "
              f"bytes={total:>10,}  target={target_bytes:>10,}  diff={diff*100:+.2f}%")
        if diff < best_diff:
            best_diff, best_crf = diff, mid
        if diff <= tol:
            return mid
        if total > target_bytes:
            # Nén chưa đủ mạnh → tăng crf_non
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    return best_crf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Frame index bắt đầu (0-based). Dùng cùng --num-frames để eval [start, start+N).")
    ap.add_argument("--num-frames", type=int, default=0,
                    help="Giới hạn số frame để debug nhanh (0 = chạy hết từ start-idx tới cuối split).")
    ap.add_argument("--gop", type=int, default=30, help="Số frame mỗi chunk encode.")
    ap.add_argument("--ckpt", type=str, default=str(SAC_CHECKPOINT))
    ap.add_argument("--methods", nargs="+", default=list(CRF_CONFIGS.keys()),
                    help="Subset trong: " + ", ".join(CRF_CONFIGS.keys()))
    ap.add_argument("--model", choices=["ccnet", "pidnet_l"], default="ccnet",
                    help="Backbone segmentation: ccnet (mặc định) hoặc pidnet_l")
    ap.add_argument("--out-dir", type=str, default=None)
    # ── Cải tiến mới ──────────────────────────────────────────────────────────
    ap.add_argument("--roi-source", choices=["pred", "gt"], default="pred",
                    help="Source của mask ROI: 'pred' = output mạng (mặc định, deployment-correct); "
                         "'gt' = từ _gtFine_4class.png (Đề xuất #1, fair so với paper-folder).")
    ap.add_argument("--no-gray", action="store_true",
                    help="Đề xuất #2: bỏ grayscale, giữ RGB khi predict seg. "
                         "⚠ CHỈ DÙNG nếu checkpoint được train với RGB input. "
                         "Checkpoint mặc định 'best_ccnet_sac.pth' train với grayscale → "
                         "bật flag này sẽ làm mIoU tụt mạnh do distribution shift.")
    ap.add_argument("--framerate", type=int, default=30,
                    help="FPS truyền cho ffmpeg. Mặc định 30 khớp pipeline gốc. "
                         "Đặt 17 nếu dùng leftImg8bit_sequence thực (Đề xuất #7).")
    ap.add_argument("--keyint", type=int, default=30,
                    help="GOP (keyint) cho codec. Mặc định 30 phù hợp pipeline slideshow "
                         "(mỗi frame test khác city, GOP dài không tận dụng được inter-pred). "
                         "Đặt 0 (codec default ~250) chỉ khi dùng video sequence thực.")
    ap.add_argument("--match-bitrate", action="store_true",
                    help="Đề xuất #8: với SA-*, binary-search crf_non để khớp bytes của H.* baseline.")
    ap.add_argument("--match-sample-gops", type=int, default=1,
                    help="Số GOP đầu dùng làm sample khi --match-bitrate (mặc định 1).")
    args = ap.parse_args()

    # Default checkpoint phụ thuộc vào model nếu user không chỉ định
    if args.ckpt == str(SAC_CHECKPOINT) and args.model == "pidnet_l":
        args.ckpt = str(PIDNET_L_CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gray = not args.no_gray
    keyint = args.keyint if args.keyint > 0 else None
    print(f"Device     : {device}")
    print(f"Split      : {args.split}")
    print(f"ROI source : {args.roi_source}")
    print(f"Use gray   : {use_gray}")
    print(f"Framerate  : {args.framerate} fps")
    print(f"Keyint     : {keyint if keyint else 'codec default'}")
    print(f"Match bw   : {args.match_bitrate}")

    if args.no_gray:
        print()
        print("  ⚠⚠⚠  --no-gray ĐANG BẬT. Checkpoint phải được train với RGB input.")
        print("       Nếu checkpoint train với grayscale (mặc định trong dataset.py),")
        print("       mIoU sẽ rất thấp (~0.5 thay vì ~0.8) do distribution shift.")
        print()

    # 1) Load model + dataset
    if args.model == "pidnet_l":
        model = build_pidnet_l(num_classes=NUM_CLASSES)
    else:
        model = build_segnet(num_classes=NUM_CLASSES, recurrence=2, pretrained=False)
    print(f"Model      : {args.model}")

    if Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            sd = (ckpt.get("model_state_dict")
                  or ckpt.get("state_dict")
                  or ckpt)
        else:
            sd = ckpt
        target = model.backbone if args.model == "pidnet_l" else model
        missing, unexpected = target.load_state_dict(sd, strict=False)
        if missing:
            print(f"  ⚠  Missing keys: {len(missing)}")
        print(f"Loaded ckpt: {args.ckpt}")
    else:
        print(f"⚠ Checkpoint {args.ckpt} không tồn tại — eval sẽ chạy với weight random / ImageNet.")
    model = model.to(device).eval()

    ds = CityscapesSAC(split=args.split, augment=False, return_original=True)
    total = len(ds)
    start = max(0, min(args.start_idx, total))
    end = total if args.num_frames <= 0 else min(start + args.num_frames, total)
    N = end - start
    if N <= 0:
        raise SystemExit(f"Range rỗng: start={start}, end={end}, total={total}.")
    print(f"Frame range: [{start}, {end}) / total={total}  → N={N} frame")

    # 2) Tiền xử lý: GT 4-class full-res, originals, mask ROI, seg trước nén
    print(f"\n[1/3] Predict seg + build ROI mask + đo seg-before cho {N} frame ...")
    originals_full: list[np.ndarray] = []
    roi_masks: list[np.ndarray] = []
    gt_4class_full: list[np.ndarray] = []
    miou_before_list: list[float] = []
    iiou_before_list: list[float] = []
    names: list[str] = []
    for j in range(N):
        i = start + j
        img_t, _lbl_t, _orig_rgb_lowres, img_path, lbl_path = ds[i]
        names.append(Path(img_path).name)
        orig_full = np.array(Image.open(img_path).convert("RGB"))
        if orig_full.shape[:2] != FRAME_HW:
            orig_full = np.array(Image.fromarray(orig_full).resize((FRAME_HW[1], FRAME_HW[0]),
                                                                   Image.BILINEAR))
        originals_full.append(orig_full)

        # GT 4-class full-res (Cityscapes label đã được preprocess sang 4 lớp)
        gt_full = np.array(Image.open(lbl_path))
        if gt_full.shape[:2] != FRAME_HW:
            gt_full = np.array(Image.fromarray(gt_full).resize((FRAME_HW[1], FRAME_HW[0]),
                                                               Image.NEAREST))
        gt_4class_full.append(gt_full)

        # Seg-before (trên frame gốc, dùng cùng pipeline use_gray)
        img_tensor_before = _frame_to_seg_tensor(orig_full, use_gray=use_gray)
        seg_before_lowres = _seg_predict(model, img_tensor_before, device)
        seg_before_full = np.array(Image.fromarray(seg_before_lowres)
                                   .resize((FRAME_HW[1], FRAME_HW[0]), Image.NEAREST))
        miou_b, _ = M.per_class_iou(seg_before_full, gt_full, NUM_CLASSES)
        iiou_b = M.iiou(seg_before_full, gt_full, ROI_CLASS_ID)
        miou_before_list.append(miou_b)
        iiou_before_list.append(iiou_b)

        # ROI mask: 'pred' = dùng seg-before; 'gt' = downsize GT về SEG_INPUT_HW
        if args.roi_source == "gt":
            gt_lowres = np.array(Image.fromarray(gt_full).resize((SEG_INPUT_HW[1], SEG_INPUT_HW[0]),
                                                                 Image.NEAREST))
            roi_masks.append(build_roi_mask(gt_lowres, target_hw=FRAME_HW))
        else:
            roi_masks.append(build_roi_mask(seg_before_lowres, target_hw=FRAME_HW))

        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{N}  (dataset idx={i})")

    mean_miou_before = float(np.mean(miou_before_list))
    print(f"\n  → mIoU(before) = {mean_miou_before*100:.2f}%  "
          f"iIoU(before) = {np.mean(iiou_before_list)*100:.2f}%")
    if mean_miou_before < 0.60:
        print(f"  ⚠ mIoU(before) thấp bất thường ({mean_miou_before*100:.1f}%).")
        print(f"    Kiểm tra: (a) checkpoint khớp với pipeline input — nếu --no-gray bật,")
        print(f"    checkpoint phải train với RGB. (b) split có nhãn không. (c) đường dẫn dataset.")

    # 3) Với từng method, nén theo GOP → đo metric
    out_dir = Path(args.out_dir) if args.out_dir else EVAL_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_frame_path = out_dir / "per_frame.csv"
    summary_path = out_dir / "summary.csv"
    per_frame_f = open(per_frame_path, "w", newline="")
    per_frame_w = csv.writer(per_frame_f)
    per_frame_w.writerow([f.name for f in FrameRecord.__dataclass_fields__.values()])

    # Lưu config eval
    (out_dir / "config.json").write_text(json.dumps({
        "split": args.split, "start_idx": start, "end_idx": end, "num_frames": N,
        "roi_source": args.roi_source, "use_gray": use_gray,
        "framerate": args.framerate, "keyint": keyint,
        "match_bitrate": args.match_bitrate, "model": args.model, "ckpt": args.ckpt,
    }, indent=2))

    summary: dict[str, dict] = {}
    baseline_bytes: dict[str, int] = {}  # 'libx264' / 'libx265' → bytes/frame baseline

    print(f"\n[2/3] Encode + đo metric per-method ...")

    # Sắp lại methods: chạy baseline H.* trước, sau đó SA-* (cần baseline_bytes cho match-bitrate)
    method_order = sorted(args.methods, key=lambda m: 0 if not m.startswith("SA-") else 1)

    for method in method_order:
        cfg = CRF_CONFIGS[method]
        codec = cfg["codec"]
        is_adaptive = cfg.get("adaptive", False)
        crf_r, crf_n = cfg["crf_roi"], cfg["crf_non"]
        is_sac = is_adaptive or (crf_r != crf_n)

        # Match-bitrate: chỉ áp dụng cho SA-* fixed, không áp dụng cho RA-CRF
        if is_sac and not is_adaptive and args.match_bitrate:
            if codec not in baseline_bytes:
                print(f"\n  ⚠ {method}: chưa có baseline {codec}, bỏ qua match-bitrate.")
            else:
                k = args.match_sample_gops * args.gop
                k = min(k, N)
                sample_frames = originals_full[:k]
                sample_masks = roi_masks[:k]
                target = baseline_bytes[codec] * k
                print(f"\n  [match-bitrate] {method}: tìm crf_non khớp {target:,} bytes "
                      f"({k} frame sample)")
                new_crf_n = _binary_search_crf(sample_frames, sample_masks, codec,
                                               crf_roi=crf_r, target_bytes=target,
                                               framerate=args.framerate, keyint=keyint)
                print(f"  → chọn crf_non = {new_crf_n} (paper config = {crf_n})")
                crf_n = new_crf_n

        if is_adaptive:
            print(f"\n=== {method}: codec={codec}, base_crf={cfg['base_crf']}, "
                  f"ΔCRF=adaptive (rule: <0.25→5, 0.25-0.43→3, >0.43→2) ===")
        else:
            print(f"\n=== {method}: codec={codec}, CRF_roi={crf_r}, CRF_non={crf_n} ===")

        recon_all: list[np.ndarray | None] = [None] * N
        # Việc 3: lưu crf thực tế dùng cho mỗi frame (phục vụ FrameRecord)
        frame_crf_roi: list[int] = [crf_r or 0] * N
        frame_crf_non: list[int] = [crf_n or 0] * N
        frame_roi_ratio: list[float] = [0.0] * N
        frame_delta_crf: list[int] = [0] * N
        bytes_total = 0
        t0 = time.time()
        for chunk in _gop_chunks(list(range(N)), args.gop):
            frames_chunk = [originals_full[i] for i in chunk]
            if is_sac:
                masks_chunk = [roi_masks[i] for i in chunk]
                if is_adaptive:
                    # Việc 1: tính roi_ratio của GOP
                    ratio = gop_roi_ratio(masks_chunk)
                    # Việc 2: chọn ΔCRF
                    delta = select_delta_crf(ratio)
                    base = cfg["base_crf"]
                    gop_crf_r = base - delta
                    gop_crf_n = base + delta
                    print(f"  GOP [{chunk[0]:4d}-{chunk[-1]:4d}]: "
                          f"roi_ratio={ratio:.3f}  ΔCRF={delta}  "
                          f"crf_roi={gop_crf_r}  crf_non={gop_crf_n}")
                else:
                    gop_crf_r, gop_crf_n = crf_r, crf_n
                    ratio, delta = 0.0, 0
                # Việc 3: encode GOP với cặp CRF tương ứng
                recon_chunk, b_r, b_n = compress_sac(
                    frames_chunk, masks_chunk, codec, gop_crf_r, gop_crf_n,
                    framerate=args.framerate, keyint=keyint)
                bytes_total += b_r + b_n
                for i in chunk:
                    frame_crf_roi[i] = gop_crf_r
                    frame_crf_non[i] = gop_crf_n
                    frame_roi_ratio[i] = ratio
                    frame_delta_crf[i] = delta
            else:
                recon_chunk, b = compress_traditional(
                    frames_chunk, codec, crf_r,
                    framerate=args.framerate, keyint=keyint)
                bytes_total += b
            for i, rec in zip(chunk, recon_chunk):
                recon_all[i] = rec
            for i in chunk[len(recon_chunk):]:
                recon_all[i] = originals_full[i]

        # Nhớ baseline để match-bitrate sau
        if not is_sac:
            baseline_bytes[codec] = bytes_total // max(N, 1)

        # 4) Metrics per-frame
        accs = {k: [] for k in ("psnr_full", "ssim_full", "psnr_roi", "ssim_roi",
                                "psnr_non", "ssim_non", "sa_psnr", "sa_ssim",
                                "miou_after", "iiou_after", "miou_drop", "iiou_drop")}

        for i in range(N):
            orig = originals_full[i]
            recon = recon_all[i]
            mask = roi_masks[i]
            non_mask = 1 - mask

            p_full = M.psnr(orig, recon)
            s_full = M.ssim(orig, recon)
            P_i, S_i = M.regional_psnr_ssim(orig, recon, mask)
            P_n, S_n = M.regional_psnr_ssim(orig, recon, non_mask)
            sa_p, sa_s = M.sa_psnr_ssim(frame_crf_roi[i], frame_crf_non[i], P_i, P_n, S_i, S_n)

            # Seg-after trên frame tái tạo
            recon_tensor = _frame_to_seg_tensor(recon, use_gray=use_gray)
            seg_after = _seg_predict(model, recon_tensor, device)
            seg_after_full = np.array(Image.fromarray(seg_after).resize((FRAME_HW[1], FRAME_HW[0]),
                                                                        Image.NEAREST))
            miou_a, _ = M.per_class_iou(seg_after_full, gt_4class_full[i], NUM_CLASSES)
            iiou_a = M.iiou(seg_after_full, gt_4class_full[i], ROI_CLASS_ID)
            miou_d = miou_before_list[i] - miou_a
            iiou_d = iiou_before_list[i] - iiou_a

            rec = FrameRecord(
                method=method,
                crf_roi=frame_crf_roi[i], crf_non=frame_crf_non[i],
                idx=start + i, name=names[i],
                psnr_full=p_full, ssim_full=s_full,
                psnr_roi=P_i, ssim_roi=S_i, psnr_non=P_n, ssim_non=S_n,
                sa_psnr=sa_p, sa_ssim=sa_s,
                miou_before=miou_before_list[i], iiou_before=iiou_before_list[i],
                miou_after=miou_a, iiou_after=iiou_a,
                miou_drop=miou_d, iiou_drop=iiou_d,
                bytes_total=bytes_total,
                roi_ratio_gop=frame_roi_ratio[i],
                delta_crf=frame_delta_crf[i],
            )
            per_frame_w.writerow([getattr(rec, k) for k in FrameRecord.__dataclass_fields__])
            per_frame_f.flush()

            accs["psnr_full"].append(p_full); accs["ssim_full"].append(s_full)
            accs["psnr_roi"].append(P_i); accs["ssim_roi"].append(S_i)
            accs["psnr_non"].append(P_n); accs["ssim_non"].append(S_n)
            accs["sa_psnr"].append(sa_p); accs["sa_ssim"].append(sa_s)
            accs["miou_after"].append(miou_a); accs["iiou_after"].append(iiou_a)
            accs["miou_drop"].append(miou_d); accs["iiou_drop"].append(iiou_d)

        elapsed = time.time() - t0
        # Crf_roi/crf_non trong summary: với adaptive lấy trung bình thực dùng
        avg_crf_r = float(np.mean(frame_crf_roi)) if is_adaptive else float(crf_r or 0)
        avg_crf_n = float(np.mean(frame_crf_non)) if is_adaptive else float(crf_n or 0)
        summary[method] = {
            "codec": codec,
            "crf_roi": avg_crf_r, "crf_non": avg_crf_n,
            "adaptive": is_adaptive,
            "avg_roi_ratio": float(np.mean(frame_roi_ratio)) if is_adaptive else None,
            "start_idx": int(start), "end_idx": int(end), "num_frames": int(N),
            "bytes_total": int(bytes_total),
            "bytes_per_frame": int(bytes_total / max(N, 1)),
            "miou_before": float(np.mean(miou_before_list)),
            "iiou_before": float(np.mean(iiou_before_list)),
            **{k: float(np.mean(v)) for k, v in accs.items()},
            "elapsed_sec": elapsed,
        }
        s = summary[method]
        ra_info = (f"  avg_roi_ratio={s['avg_roi_ratio']:.3f}" if is_adaptive else "")
        print(f"  → SA-PSNR={s['sa_psnr']:.3f} dB  SA-SSIM={s['sa_ssim']:.4f}  "
              f"mIoU(before/after/Δ)={s['miou_before']*100:.2f}/{s['miou_after']*100:.2f}/"
              f"{s['miou_drop']*100:+.2f}%  "
              f"iIoU(before/after/Δ)={s['iiou_before']*100:.2f}/{s['iiou_after']*100:.2f}/"
              f"{s['iiou_drop']*100:+.2f}%  "
              f"bytes/frame={s['bytes_per_frame']:,}  ({elapsed:.1f}s){ra_info}")

    per_frame_f.close()
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + list(next(iter(summary.values())).keys()))
        for m, d in summary.items():
            w.writerow([m] + list(d.values()))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[3/3] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
