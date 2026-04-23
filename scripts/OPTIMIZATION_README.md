# SAC-X265 CRF Optimization Suite

Complete pipeline for optimizing H.265 compression parameters for Semantic-Aware video compression.

## Files

### 1. `optimize_sac_crf.py` - Main Optimization Engine

**Purpose**: Grid-search CRF (Constant Rate Factor) combinations for ROI and non-ROI regions.

**What it does**:
- Takes input frame images from Cityscapes validation set
- For each CRF combination (e.g., CRF_ROI=25, CRF_NON=32):
  - Encodes ROI stream and non-ROI stream with different CRF values
  - Merges decoded streams back together
  - Compares against traditional single-stream baseline
  - Computes quality metrics (PSNR, SSIM, SA-PSNR, SA-SSIM)
  - Measures bitrate consumption
  - Ranks by objective: `score = Δ SA-PSNR - λ × Δ Bitrate`
- Outputs structured metrics CSV and per-combo video artifacts

**Usage**:
```bash
python optimize_sac_crf.py \
  --num-frames 20 \
  --roi-crf-list 20,23,25 \
  --non-crf-list 30,32,35 \
  --preset medium \
  --objective-lambda 0.4
```

**Key Arguments**:
- `--num-frames`: How many test frames to use (small = fast, large = accurate)
- `--roi-crf-list`: Comma-separated CRF values for ROI region (lower = higher quality)
- `--non-crf-list`: Comma-separated CRF values for non-ROI region
- `--objective-lambda`: Weight for bitrate penalty (λ in objective function)
  - λ=0.2: Prioritize quality over bitrate
  - λ=0.4: Balanced (default)
  - λ=0.8: Prioritize efficiency over quality
- `--keep-artifacts`: Keep temporary frame files (default: remove)

**Output**:
```
outputs/optimization/<TIMESTAMP>/
├── metrics/
│   ├── crf_optimization_summary.csv      ← Main results table
│   ├── metrics_roi*_non*.csv             ← Per-frame details for each config
│   ├── best_config_report.txt            ← Text summary
│   └── plots/
│       ├── rd_curve_psnr.png             ← Rate-Distortion plots
│       ├── pareto_trade_off.png          ← Trade-off analysis
│       ├── config_heatmap_*.png          ← Parameter sensitivity
│       └── ... (7 PNG files total)
├── combos/
│   ├── roi20_non30/
│   │   ├── roi.mp4
│   │   ├── nonroi.mp4
│   │   ├── sac_x265.mp4
│   │   └── traditional_x265.mp4
│   └── ... (one folder per CRF config)
└── EXECUTIVE_SUMMARY.md
```

**Example Workflows**:

Initial quick test (3×3 grid):
```bash
python optimize_sac_crf.py --num-frames 20 --roi-crf-list 20,23,25 --non-crf-list 30,32,35
```

Expanded grid (5×4 = 20 configs):
```bash
python optimize_sac_crf.py --num-frames 50 --roi-crf-list 20,22,24,25,27 --non-crf-list 30,31,32,33,35
```

High-quality validation (100 frames):
```bash
python optimize_sac_crf.py --num-frames 100 --roi-crf-list 23,24,25,26 --non-crf-list 30,31,32,33,34
```

Efficiency-focused (λ=0.8):
```bash
python optimize_sac_crf.py --num-frames 40 --objective-lambda 0.8 --roi-crf-list 24,25,26 --non-crf-list 33,34,35
```

---

### 2. `visualize_optimization_results.py` - Visualization Generator

**Purpose**: Generate publication-ready plots from optimization CSV results.

**What it does**:
- Reads `crf_optimization_summary.csv` from optimize_sac_crf.py output
- Generates 7 high-quality PNG plots (300 DPI, publication-ready)
- Creates plots directory if not exists
- Generates plots in parallel for speed

**Usage**:
```bash
python visualize_optimization_results.py <paths_to_summary.csv>
```

**Examples**:
```bash
# Generate plots from specific run
python visualize_optimization_results.py outputs/optimization/20260404_154235/metrics/crf_optimization_summary.csv

# Or with custom output directory
python visualize_optimization_results.py \
  outputs/optimization/20260404_154235/metrics/crf_optimization_summary.csv \
  --output-dir outputs/my_thesis_figures/
```

**Output Plots**:
1. `rd_curve_psnr.png` - Bitrate vs SA-PSNR trade-off
2. `rd_curve_ssim.png` - Bitrate vs SA-SSIM trade-off
3. `pareto_trade_off.png` - Quality-efficiency Pareto front
4. `config_heatmap_sapsnr.png` - CRF combinations matrix (SA-PSNR)
5. `config_heatmap_delta.png` - CRF combinations matrix (Quality gain)
6. `metric_comparison_bars.png` - Multi-panel bar charts (Top configs vs baseline)
7. `score_ranking.png` - Ranked configurations by objective score

**Use These Plots In Your Thesis**:
- **Section 4.1 (Configuration Space)**: Use `config_heatmap_sapsnr.png` 
- **Section 4.2 (Rate-Distortion)**: Use `rd_curve_psnr.png`
- **Section 4.3 (Optimization Results)**: Use `pareto_trade_off.png` (main result!)
- **Section 4.4 (Comparison)**: Use `metric_comparison_bars.png`
- **Appendix**: Include all plots for completeness

---

## Recommended 1-Month Workflow

### Week 1: Baseline & Quick Validation
```bash
# Run initial 3×3 grid on 20 frames
python optimize_sac_crf.py --num-frames 20 --roi-crf-list 20,23,25 --non-crf-list 30,32,35

# Generate plots from results
python visualize_optimization_results.py outputs/optimization/*/metrics/crf_optimization_summary.csv
```

### Week 2: Expand & Refine
```bash
# Run 5×4 grid on 50 frames
python optimize_sac_crf.py --num-frames 50 --roi-crf-list 20,22,24,25,27 --non-crf-list 30,31,32,33,35

# Re-generate plots
python visualize_optimization_results.py outputs/optimization/*/metrics/crf_optimization_summary.csv
```

### Week 3: Sensitivity Analysis
```bash
# Test different λ values to trace Pareto frontier
for lambda in 0.2 0.4 0.6 0.8; do
  python optimize_sac_crf.py --num-frames 50 --objective-lambda $lambda \
    --roi-crf-list 22,24,25,26 --non-crf-list 30,32,34
done

# Compare results across different λ runs
```

### Week 4: Statistical Validation
```bash
# Run on full/large dataset for confidence intervals
python optimize_sac_crf.py --num-frames 200 --roi-crf-list 23,24,25,26,27 --non-crf-list 30,31,32,33,34,35

# Compute stats manually or modify script to add confidence intervals
```

---

## Performance Notes

- **GPU Memory**: ~1.2 GB per frame during encoding (RTX 1650)
- **Encoding Speed**: ~3-5 seconds per frame (medium preset, all 3 streams)
- **Typical Runtime**:
  - 20 frames, 9 configs (3×3): ~15 minutes
  - 50 frames, 20 configs (5×4): ~60 minutes
  - 100 frames, 30 configs: ~3 hours

- **Disk Space**: ~100 MB per frame set (temporary), ~50 KB per config metrics CSV

---

## Objective Function Explained

```
score = Δ SA-PSNR_vs_Trad - λ × Δ Bitrate_vs_Trad
```

- Left term: Quality improvement (how much better than traditional)
- Right term: Bitrate cost (higher λ = more penalty for bitrate)
- λ = 0.4 (default) means: 1 Mbps bitrate increase is worth ~0.4 dB quality penalty

**Choosing λ**:
- λ=0.2: "I want best quality regardless of bitrate" → Lower CRF values
- λ=0.4: "Good balance" (recommended) → CRF 25/32
- λ=0.8: "I want smallest bitrate possible" → Higher CRF values

---

## Troubleshooting

**Error**: `ffmpeg not found`
- Solution: Install ffmpeg with libx265 support
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # macOS
  brew install ffmpeg --with-x265
  ```

**Error**: `Model not found: models/best_pidnet.pth`
- Solution: Train the segmentation model first (see train_segmentation.py)

**Error**: `Image directory not found`
- Solution: Ensure Cityscapes data is prepared in `data/gt_4class/leftImg8bit_trainvaltest/leftImg8bit/val/`

**Slow Performance**:
- Reduce `--num-frames` for testing
- Use `--preset fast` instead of `medium` (trades quality for speed)
- Reduce CRF search space (fewer --roi-crf-list and --non-crf-list values)

---

## Integration with Your Thesis

All plots and data are thesis-ready. Example LaTeX integration:

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.9\textwidth]{../outputs/optimization/20260404_154235/metrics/plots/pareto_trade_off.png}
  \caption{SAC-X265 CRF optimization showing quality-efficiency trade-offs.}
  \label{fig:pareto_front}
\end{figure}

\begin{table}[h]
  \centering
  \input{../outputs/optimization/20260404_154235/metrics/crf_optimization_summary.tex}
  \caption{CRF configuration optimization results ranked by objective score.}
  \label{tab:crf_optimization}
\end{table}
```

---

## References

- **Original SAC Paper**: "Semantic-Aware Video Compression for Automotive Cameras"
- **H.265/HEVC**: https://github.com/videolan/x265
- **Rate-Distortion Theory**: https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory

---

**Last Updated**: 2026-04-04  
**Author Notes**: Optimized for 1-month thesis sprint with RTX 1650 GPU
