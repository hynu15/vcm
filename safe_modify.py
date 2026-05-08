with open('scripts/run_segmentation_rd_pipeline.py', 'r') as f:
    text = f.read()

import re

# 1. Update REQUESTED_POINTS
old_pts = """REQUESTED_POINTS: Sequence[OperatingPoint] = (
    OperatingPoint("23/32", 23, 32),
    OperatingPoint("25/32", 25, 32),
    OperatingPoint("25/30", 25, 30),
    OperatingPoint("25/35", 25, 35),
)"""
new_pts = """REQUESTED_POINTS: Sequence[OperatingPoint] = (
    OperatingPoint("22", 20, 36),
    OperatingPoint("27", 25, 41),
    OperatingPoint("32", 30, 46),
    OperatingPoint("37", 35, 51),
)"""
text = text.replace(old_pts, new_pts)

# 2. Update encode_two_streams signature and logic
text = text.replace(
"""def encode_two_streams(
    frame_dir: Path,
    output_dir: Path,
    fps: int,
    crf_roi: int,
    crf_non: int,
    preset: str,
) -> Tuple[Path, Path, Path, Path]:""",
"""def encode_two_streams(
    frame_dir: Path,
    output_dir: Path,
    fps: int,
    crf_roi: int,
    crf_non: int,
    crf_trad: int,
    preset: str,
) -> Tuple[Path, Path, Path, Path]:""")

text = text.replace("crf_trad = int(round((crf_roi + crf_non) / 2.0))", "")

# 3. Update the call
old_call = """        roi_video, non_video, sac_video, trad_video = encode_two_streams(
            frame_dir=frame_dir,
            output_dir=combo_dir,
            fps=args.fps,
            crf_roi=op.crf_roi,
            crf_non=op.crf_non,
            preset=args.preset,
        )"""
new_call = """        roi_video, non_video, sac_video, trad_video = encode_two_streams(
            frame_dir=frame_dir,
            output_dir=combo_dir,
            fps=args.fps,
            crf_roi=op.crf_roi,
            crf_non=op.crf_non,
            crf_trad=int(op.label),
            preset=args.preset,
        )"""
text = text.replace(old_call, new_call)

# 4. Details
text = text.replace("crf_trad = int(round((op.crf_roi + op.crf_non) / 2.0))", "crf_trad = int(op.label)")

with open('scripts/run_segmentation_rd_pipeline.py', 'w') as f:
    f.write(text)
