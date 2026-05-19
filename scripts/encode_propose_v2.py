import argparse
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_segmentation import load_segmentation_model


def get_qp_delta(class_id):
    """
    Quy định mức độ bù trừ QP theo class.
    Class 0: ROI (đường/xe/người) -> nén ít hơn (QP thấp hơn)
    Class 1: non_ROI (sky/construction/nature) -> nén mạnh hơn (QP cao hơn)
    """
    mapping = {
        0: -4,  # ROI (important)
        1:  5,  # non_ROI (unimportant)
    }
    return mapping.get(class_id, 0)


def build_adaptive_qp_file(seg_maps, base_qp, output_path):
    """
    Sinh ra file qpfile (định dạng Frame-level của x265)
    Dựa trên diện tích trung bình của từng class trên frame.
    """
    qp_per_frame = []
    
    for seg in seg_maps:
        unique, counts = np.unique(seg, return_counts=True)
        
        weighted_qp = 0
        total_pixels = seg.size
        
        for val, count in zip(unique, counts):
            weighted_qp += count * (base_qp + get_qp_delta(val))
            
        avg_qp = int(round(weighted_qp / total_pixels))
        avg_qp = max(0, min(avg_qp, 51)) # Kẹp trong khoảng 0-51
        qp_per_frame.append(avg_qp)
        
    with open(output_path, 'w') as f:
        # x265 qpfile format: <frameID> <qp> <frametype>
        for i, qp in enumerate(qp_per_frame):
            # Cứ 30 frame gán là I-frame, còn lại P-frame để đơn giản
            ftype = 'I' if i % 30 == 0 else 'P'
            f.write(f'{i} {qp} {ftype}\n')
            
    return qp_per_frame


def process_single_stream_sac(
    input_video, output_video, model, device, base_qp, method="blur"
):
    """
    Hai phương pháp tối ưu:
    1. method="qpfile": Temporal SAC - Chỉnh QP cả frame tùy theo ratio ROI
    2. method="blur": Spatial SAC ngầm định - Làm mờ Background để tiết kiệm Bandwidth
    """
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tmp_dir = tempfile.mkdtemp(prefix="sac_v2_")
    
    seg_maps = []
    frame_idx = 0
    
    print(f"🔄 Processing frames using method='{method}'...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Inference
        img_pil = Image.fromarray(frame_rgb)
        inp = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)
        mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        seg_maps.append(mask)
        
        if method == "blur":
            # SPATIAL approach: Blur the non-ROI regions heavily
            roi_mask = (mask == 0).astype(np.uint8)
            roi_3 = np.repeat(roi_mask[..., None], 3, axis=2)
            
            blurred = cv2.GaussianBlur(frame_bgr, (51, 51), 20)
            hybrid = np.where(roi_3 == 1, frame_bgr, blurred).astype(np.uint8)
            cv2.imwrite(os.path.join(tmp_dir, f"frame_{frame_idx:04d}.png"), hybrid)
            
        elif method == "qpfile":
            # TEMPORAL approach: Save raw frames, encoding will use qpfile
            cv2.imwrite(os.path.join(tmp_dir, f"frame_{frame_idx:04d}.png"), frame_bgr)
            
        frame_idx += 1
        
    cap.release()
    
    # Bắt đầu Encode
    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%04d.png"),
        "-c:v", "libx265",
        "-preset", "medium",
        "-pix_fmt", "yuv420p"
    ]
    
    if method == "qpfile":
        qpfile_path = os.path.join(tmp_dir, "qpfile.txt")
        qp_list = build_adaptive_qp_file(seg_maps, base_qp, qpfile_path)
        print(f"📊 Generate QPFile with Avg QP: {np.mean(qp_list):.2f}")
        cmd.extend([
            "-x265-params", f"qpfile={qpfile_path}:aq-mode=0:keyint=30"
        ])
    else:
        cmd.extend(["-crf", str(base_qp)])
        
    cmd.append(output_video)
    
    print(f"🎬 Running FFmpeg...")
    subprocess.run(cmd, check=True)
    print(f"✅ Saved to {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single-Stream SAC V2")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-qp", type=int, default=28)
    parser.add_argument("--method", choices=["qpfile", "blur"], default="blur")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_pidnet.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_ccnet.pth")
        
    model, _ = load_segmentation_model(model_path, device=device, num_classes=2)
    
    process_single_stream_sac(args.video, args.output, model, device, args.base_qp, args.method)