# CLAUDE.md

## Project Overview

This repository implements a Semantic-Aware Video Compression (SAC) pipeline for automotive camera videos. The main idea is to use semantic segmentation to separate each frame into two streams:

1. **ROI stream**: region of interest, containing important road-related objects and areas.
2. **non-ROI stream**: background or less important regions.

The two streams are encoded separately with different compression qualities. ROI is compressed with better quality, while non-ROI is compressed more aggressively. After transmission or storage, the two decoded streams are combined to reconstruct the final video frame.

This project is used as a baseline for experiments on semantic-aware video coding for machine vision tasks, especially segmentation-aware evaluation on automotive datasets such as Cityscapes and KITTI-STEP.

## Main Goal for Claude

When modifying this repository, prioritize:

* Keeping the original SAC pipeline correct and reproducible.
* Making path configuration, dataset organization, compression scripts, and evaluation scripts easier to run.
* Supporting experiments that compare:

  * traditional H.264/AVC or H.265/HEVC compression,
  * SAC with CCNet,
  * SAC with PIDNet-L or other segmentation models.
* Preserving compatibility with the original repository structure unless explicitly asked to refactor.

## Repository Structure

Important directories:

```text
Codes/
├── CCNet/          # Semantic segmentation and two-stream generation using CCNet
├── EfficientPS/    # Alternative segmentation model for KITTI-STEP
├── compression/    # FFmpeg compression commands and related scripts
├── Eva/            # Evaluation scripts
└── Tools/          # Utility scripts
```

The common SAC workflow is:

```text
Input frames
    ↓
Semantic segmentation
    ↓
ROI / non-ROI mask generation
    ↓
Two-stream frame generation
    ↓
Separate video compression using FFmpeg
    ↓
Decode compressed streams
    ↓
Combine ROI and non-ROI streams
    ↓
Evaluate reconstructed frames
```

## Core Pipeline

### Step 1: Two-stream generation

For the CCNet-based pipeline, the original workflow is:

```bash
cd Codes/CCNet
python Rename_Rescale.py
python test.py
python TwoStream_generate.py
```

Expected behavior:

* `Rename_Rescale.py` prepares or resizes input frames.
* `test.py` runs semantic segmentation and generates segmentation masks.
* `TwoStream_generate.py` uses the segmentation result to generate ROI and non-ROI streams.

When editing these scripts, do not change the semantic meaning of ROI and non-ROI unless explicitly requested.

### Step 2: Compression

Compression is handled using FFmpeg with x264 or x265.

Typical parameters that may be changed during experiments:

```text
-codec / -c:v       libx264 or libx265
-crf                constant rate factor
-framerate          input frame rate
-preset             encoder speed-quality tradeoff
-g / keyint         GOP length
-pix_fmt            usually yuv420p
```

For SAC experiments, use different CRF values for the two streams:

```text
C_ROI  < C_nonROI
```

A lower CRF means higher reconstruction quality. Therefore, ROI should use a lower CRF than non-ROI.

Example experiment setting:

```text
Baseline: one full-frame stream with CRF = 22, 25, 28, 31, 34

SAC: two streams
C_ROI    = C_base - 3
C_nonROI = C_base + 3
```

Example CRF pairs:

```text
(19, 25), (22, 28), (25, 31), (28, 34), (31, 37)
```

### Step 3: Stream combination

After compression and decoding, ROI and non-ROI frames are combined to reconstruct the final output frame.

The combination should follow this principle:

```python
reconstructed = roi_decoded + nonroi_decoded
```

Use saturated addition if using OpenCV:

```python
reconstructed = cv2.add(roi_decoded, nonroi_decoded)
```

This is valid because ROI and non-ROI streams should not contain non-zero pixels at the same spatial positions, except for minor compression artifacts.

### Step 4: Evaluation

Evaluation scripts are stored in:

```bash
Codes/Eva/
```

Evaluation should include, when available:

```text
PSNR
SSIM
SA-PSNR
SA-SSIM
mIoU
iIoU
bitrate
encoding time
decoding time
segmentation inference time
```

For thesis experiments, prioritize semantic-aware metrics because the purpose of the project is video compression for machine vision, not only human visual quality.

## Thesis-specific Extension

This repository may be extended for a robotics or automotive vision thesis with the following modifications:

1. Replace or compare the original CCNet segmentation model with PIDNet-L.
2. Generate four-class semantic masks:

   * ROI
   * sky
   * construction
   * nature
3. Convert the semantic segmentation mask into a binary ROI/non-ROI mask.
4. Quantize the mask into block-level maps using a 16×16 grid.
5. Encode ROI and non-ROI streams separately using FFmpeg.
6. Compare traditional compression against semantic-aware compression.
7. Evaluate both image quality and downstream segmentation performance.

When implementing PIDNet-L, keep it modular. Do not hard-code it inside CCNet-specific scripts unless the user requests a quick prototype.

Recommended structure for extension:

```text
Codes/
├── CCNet/
├── PIDNet/
├── compression/
├── Eva/
└── Tools/
```

or:

```text
Codes/
├── segmentation/
│   ├── ccnet/
│   └── pidnet/
├── stream_generation/
├── compression/
├── reconstruction/
└── evaluation/
```

Prefer the first structure if the goal is minimal modification. Prefer the second structure if the user wants a cleaner thesis-ready implementation.

## Important Implementation Rules

### Path handling

Avoid hard-coded absolute paths such as:

```python
/home/user/...
/content/drive/...
D:/...
```

Use `argparse` whenever possible:

```python
parser.add_argument("--input_dir", required=True)
parser.add_argument("--mask_dir", required=True)
parser.add_argument("--roi_dir", required=True)
parser.add_argument("--nonroi_dir", required=True)
parser.add_argument("--output_dir", required=True)
```

If a default path is necessary, place it near the top of the script and make it easy to modify.

### File naming

Use consistent frame naming:

```text
frame_00000.png
frame_00001.png
frame_00002.png
```

or:

```text
000000.png
000001.png
000002.png
```

Do not mix naming formats inside the same pipeline.

### Image size

Cityscapes original resolution is usually:

```text
2048×1024
```

Common resized input for segmentation:

```text
1024×512
```

When generating masks, always ensure the mask is resized back to the same resolution as the frame being encoded.

Use nearest-neighbor interpolation for label masks:

```python
cv2.INTER_NEAREST
```

Do not use bilinear interpolation for class label masks because it creates invalid class values.

### Mask logic

For binary ROI mask:

```python
roi_mask = np.isin(segmentation_mask, roi_class_ids)
nonroi_mask = ~roi_mask
```

For multiplication with RGB frames:

```python
roi_frame = frame * roi_mask[..., None]
nonroi_frame = frame * nonroi_mask[..., None]
```

Make sure masks are either boolean or normalized to 0/1 before multiplication.

### 16×16 block-level mask

When using macroblock-level ROI selection:

* Divide the mask into 16×16 blocks.
* If a block contains at least one ROI pixel, mark the entire block as ROI.
* This protects small objects such as pedestrians, signs, poles, and vehicles.

Expected rule:

```python
if np.any(block == ROI):
    block_mask = ROI
else:
    block_mask = nonROI
```

Do not use average pooling unless explicitly requested, because it may remove small ROI objects.

## Coding Style

Use clear Python code with:

* meaningful variable names,
* small functions,
* comments for image/mask shape transformations,
* explicit input and output directories,
* progress logs for long-running scripts.

Prefer this style:

```python
def generate_two_streams(frame_dir, mask_dir, roi_dir, nonroi_dir, block_size=16):
    ...
```

Avoid large scripts where all logic is placed directly under the global scope.

Use:

```python
if __name__ == "__main__":
    main()
```

for executable scripts.

## Reproducibility Rules

When adding or modifying experiments, always record:

```text
dataset name
number of frames
image resolution
segmentation model
checkpoint path
codec
CRF values
GOP length
frame rate
output bitrate
evaluation metrics
hardware
```

For thesis experiments, save results into CSV files such as:

```text
results_x264.csv
results_x265.csv
results_ccnet.csv
results_pidnet.csv
```

Recommended CSV columns:

```text
method,codec,crf_roi,crf_nonroi,bitrate,psnr,ssim,sa_psnr,sa_ssim,miou,iiou,fps
```

## Safety Rules for Code Changes

Before modifying code:

1. Identify which part of the pipeline the file belongs to:

   * segmentation,
   * mask generation,
   * compression,
   * reconstruction,
   * evaluation.
2. Preserve existing behavior unless the user asks for a change.
3. Avoid deleting old scripts. If a script is messy but working, create a cleaner new script instead.
4. Do not remove citations, README information, or license files.
5. Do not commit datasets, generated frames, videos, checkpoints, or large result files.

Add large files to `.gitignore`:

```gitignore
*.mp4
*.avi
*.mkv
*.pth
*.pt
*.ckpt
*.onnx
*.npy
*.npz
datasets/
data/
outputs/
results/
frames/
checkpoints/
```

## Common Commands

### Run CCNet pipeline

```bash
cd Codes/CCNet
python Rename_Rescale.py
python test.py
python TwoStream_generate.py
```

### Run compression

Use scripts in:

```bash
Codes/compression/
```

or use FFmpeg directly.

Example x264:

```bash
ffmpeg -framerate 30 -i frame_%05d.png -c:v libx264 -crf 22 -preset medium -pix_fmt yuv420p output_x264.mp4
```

Example x265:

```bash
ffmpeg -framerate 30 -i frame_%05d.png -c:v libx265 -crf 22 -preset medium -pix_fmt yuv420p output_x265.mp4
```

### Combine two streams

```bash
python combine.py
```

### Run evaluation

```bash
cd Codes/Eva
python <evaluation_script>.py
```

## Preferred Improvements

When asked to improve the repository, prioritize these tasks:

1. Add command-line arguments to scripts.
2. Create a unified configuration file.
3. Make input/output paths clear.
4. Add automatic folder creation.
5. Add logging for each pipeline stage.
6. Add CSV result export.
7. Add support for PIDNet-L.
8. Add batch evaluation over multiple CRF pairs.
9. Add separate evaluation for ROI and non-ROI.
10. Add scripts for thesis figures and tables.

## Do Not Do

Do not:

* change ROI/non-ROI definitions silently,
* use bilinear interpolation for segmentation label masks,
* mix RGB and BGR without clear conversion,
* overwrite original frames,
* overwrite original segmentation masks,
* commit large generated videos or datasets,
* assume all datasets have the same label format,
* report metrics without recording the exact CRF and codec settings.

## Notes for Thesis Writing

When generating explanations for the thesis, use Vietnamese academic writing style.

Preferred terms:

```text
semantic-aware video compression  -> nén video nhận thức ngữ nghĩa
region of interest                -> vùng quan tâm
region out of interest            -> vùng ngoài quan tâm
two-stream compression             -> nén hai luồng
semantic segmentation              -> phân đoạn ngữ nghĩa
macroblock-level mask              -> mặt nạ cấp khối
constant rate factor               -> hệ số chất lượng không đổi CRF
```

When explaining the method, emphasize:

* SAC is designed for machine vision, not only human visual perception.
* ROI should preserve road stakeholders and important driving-related regions.
* non-ROI can be compressed more strongly to reduce bitrate.
* PIDNet-L can improve real-time segmentation and boundary quality compared with heavier attention-based segmentation models.
* Evaluation should include both traditional quality metrics and semantic-aware metrics.

## Expected Output Quality

Any new code should be:

* executable,
* readable,
* reproducible,
* easy to configure,
* suitable for thesis experiments.

Any new explanation should be:

* technically correct,
* concise,
* written in formal Vietnamese if used for the thesis,
* consistent with the SAC pipeline.
