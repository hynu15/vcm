# BD-Rate / BD-Accuracy Pipeline: Semantic-Aware Video Compression Evaluation

Đây là complete pipeline để tính **Bjøntegaard Delta** metrics (BD-Rate, BD-Accuracy) cho đánh giá video compression.

## 📚 Tl;DR - Quick Start

```bash
# Chạy demo trên dữ liệu sample
cd /home/huy/sac_project/scripts
conda activate sac
python bd_demo.py

# Hoặc chạy trên dữ liệu riêng của bạn
python compute_bd_pipeline.py \
    --baseline-csv baseline_rd.csv \
    --propose-csv propose_rd.csv \
    --metric miou \
    --output-dir results/
```

## 🎯 Objectives

Pipeline tính các chỉ số chuẩn trong đánh giá video compression:

- **BD-Rate (%)**: Ở cùng Accuracy, Propose cần ít bitrate hơn bao nhiêu % so với Baseline
  - Âm = Propose tốt hơn ✅ (tiết kiệm bitrate)
  - Dương = Propose tệ hơn ❌ (tốn thêm bitrate)

- **BD-Accuracy**: Ở cùng Bitrate, Propose có Accuracy cao hơn bao nhiêu (tuyệt đối, không %)
  - Dương = Propose tốt hơn ✅
  - Âm = Propose tệ hơn ❌

## 📋 Input Format

CSV files với format:
```
qp,bitrate_kbps,accuracy_mean
22,3000,0.835
27,2000,0.828
32,1200,0.815
37,800,0.795
```

**Bắt buộc columns:**
- `qp`: Độ lượng tử hóa (22, 27, 32, 37 là chuẩn bộ 4 điểm)
- `bitrate_kbps`: Bitrate sau compress (đơn vị: kbps)
- `accuracy_mean`: Độ chính xác (mIoU/mAP/MOTA tùy task, đã aggregate từ tất cả frames)

## 🔄 Pipeline Steps

### STEP 0: Chuẩn hóa Input
Nếu dữ liệu gốc là frame-wise (mỗi frame 1 row):
```python
# Aggregate: per-frame mean → per-video mean → dataset mean
df_aggregated = df.groupby(['method', 'qp']).agg({
    'bitrate_kbps': 'mean',
    'accuracy': 'mean',
}).reset_index()
```

### STEP 1-2: Validate RD Points
- Kiểm tra monotonicity: QP ↑ → bitrate ↓, bitrate ↑ → accuracy ↑
- Kiểm tra overlap: vùng Rate chồng lấp giữa Baseline & Propose
- Nếu không có overlap → BD metrics không có ý nghĩa ❌

### STEP 3-4: Tính BD Metrics (Bjøntegaard Interpolation)
1. Nội suy curves trên **log(bitrate)** vs **accuracy** (PCHIP - Piecewise Cubic Hermite)
2. Tích phân trên vùng overlap:
   - **BD-Rate**: `∫[rate_propose/rate_baseline - 1] dacc` (normalized by acc range)
   - **BD-Accuracy**: `∫[acc_propose - acc_baseline] drate` (normalized by rate range)

### STEP 5: Xuất Bảng
- **CSV**: Tiện lợi cho post-processing
- **Markdown**: Dễ đọc, dùng cho tài liệu
- **LaTeX**: Sẵn sàng paste vào paper
- **JSON**: Structured data cho automation

### STEP 6: Plot RD Curve
- X-axis: Bitrate (linear hoặc log scale)
- Y-axis: Accuracy
- Baseline: solid line, blue
- Propose: dashed line, orange
- Annotation: QP values trên mỗi điểm

## 📊 Output Examples

### Console Output
```
===============================================================
📊 BD-RATE / BD-ACCURACY RESULTS
===============================================================
 QP Baseline Acc Baseline Rate Propose Acc Propose Rate
 22       0.8350            3000      0.8450            2500
 27       0.8280            2000      0.8380            1650
 32       0.8150            1200      0.8280             950
 37       0.7950             800      0.8100             600

------------------------------------------------------------
BD-Rate: -49.87%
  ✅ Propose tốt hơn: tiết kiệm 49.87% bitrate ở cùng mIoU
BD-Accuracy: +0.0173 (tuyệt đối)
  ✅ Propose tốt hơn: +0.0173 mIoU ở cùng bitrate
===============================================================
```

### CSV Table
```csv
QP,Baseline Acc,Baseline Rate (kbps),Propose Acc,Propose Rate (kbps)
22,0.8350,3000,0.8450,2500
27,0.8280,2000,0.8380,1650
32,0.8150,1200,0.8280,950
37,0.7950,800,0.8100,600
```

### LaTeX Table
```latex
\begin{table}[ht!]
  \caption{Rate-Distortion Comparison}
  \begin{tabular}{ccccc}
    \toprule
    QP & Baseline Acc & Baseline Rate & Propose Acc & Propose Rate \\
    \midrule
    22 & 0.8350 & 3000 & 0.8450 & 2500 \\
    ...
  \end{tabular}
\end{table}
```

## 🚀 Usage Examples

### Example 1: SAC vs Traditional
```bash
python compute_bd_pipeline.py \
    --baseline-csv sac_traditional_baseline.csv \
    --propose-csv sac_improved_propose.csv \
    --metric miou \
    --output-dir results/sac_vs_trad \
    --title "SAC Improved vs Traditional H.265"
```

### Example 2: Multi-Task Evaluation
Run riêng cho từng task:

```bash
# Segmentation
python compute_bd_pipeline.py \
    --baseline-csv baseline_seg.csv \
    --propose-csv propose_seg.csv \
    --metric miou \
    --output-dir results/segmentation

# Detection
python compute_bd_pipeline.py \
    --baseline-csv baseline_det.csv \
    --propose-csv propose_det.csv \
    --metric map \
    --output-dir results/detection

# Tracking
python compute_bd_pipeline.py \
    --baseline-csv baseline_track.csv \
    --propose-csv propose_track.csv \
    --metric mota \
    --output-dir results/tracking
```

### Example 3: Demo on Sample Data
```bash
python bd_demo.py
# Output: outputs/bd_demo/
```

## 📁 Output Directory Structure

```
results/
├── bd_metrics.json           # BD scores
├── rd_points.csv             # Normalized RD data
├── rd_curve.png              # 2D plot
└── tables/
    ├── bd_results_table.csv  # CSV format
    ├── bd_results_table.md   # Markdown
    ├── bd_results_table.tex  # LaTeX
    └── bd_summary.json       # Summary
```

## ⚙️ Module Details

### `compute_rd_points.py`
- Normalize input data
- Validate monotonicity & overlap
- Usage: `python compute_rd_points.py --input raw.csv --output normalized.csv`

### `compute_bd.py`
- PCHIP interpolation on log(rate)
- Bjøntegaard delta calculation
- Usage: `python compute_bd.py --rd-csv normalized.csv --output bd_metrics.json`

### `plot_rd.py`
- Plot RD curves
- Annotate QP values
- Usage: `python plot_rd.py --rd-csv normalized.csv --output rd_curve.png`

### `generate_bd_tables.py`
- Xuất bảng (CSV, MD, LaTeX, JSON)
- Usage: `python generate_bd_tables.py --rd-csv normalized.csv --bd-json bd_metrics.json --output-dir tables/`

### `compute_bd_pipeline.py`
- Complete pipeline wrapper
- Usage: `python compute_bd_pipeline.py --baseline-csv bl.csv --propose-csv pr.csv --output-dir results/`

### `bd_demo.py`
- Demo on synthetic data
- Tự động kiểm tra toàn bộ pipeline
- Usage: `python bd_demo.py`

## 🔍 Common Issues & Troubleshooting

### Issue: "Rate ranges do not overlap"
**Cause**: Vùng bitrate của Baseline và Propose không chồng lấp
**Solution**:
  - Chọn QP range rộng hơn (ví dụ thêm QP=18, 42)
  - Hoặc sử dụng phương pháp extrapolation (cẩn thận!)

### Issue: "Validation warnings: Bitrate increases but accuracy NOT increases"
**Cause**: Dữ liệu bị noisy hoặc không đúng chuẩn RD curve
**Solution**:
  - Check xem có phải frame-wise data cần aggregate
  - Ensure bitrate tính từ encoded file size chính xác
  - Nếu là test data, có thể ignore warning

### Issue: "No overlap detected"
**Cause**: Lần đầu tiên gặp phải — hiếm khi xảy ra nếu QP range chuẩn
**Solution**:
  - Verify QP values trong CSV
  - Check bitrate-to-QP relationship (hình học)

## 📖 Mathematical Background

### Bjøntegaard Delta (BD-Rate)
Given two RD curves `(r_1, acc_1)` and `(r_2, acc_2)`:

$$BD\text{-}Rate = \frac{1}{acc_{max} - acc_{min}} \int_{acc_{min}}^{acc_{max}} \left( \frac{r_2(acc)}{r_1(acc)} - 1 \right) 100\% \, dacc$$

Simplified với linear interpolation trên `log(rate)` vs `accuracy`:

$$BD\text{-}Rate \approx \frac{1}{N} \sum_{i=1}^{N} \left( \frac{r_2(acc_i)}{r_1(acc_i)} - 1 \right) 100\%$$

### Bjøntegaard Delta (BD-Accuracy)
$$BD\text{-}Accuracy = \frac{1}{\log(r_{max}) - \log(r_{min})} \int_{\log(r_{min})}^{\log(r_{max})} \left( acc_2(log r) - acc_1(log r) \right) d(log r)$$

Simplified:
$$BD\text{-}Accuracy \approx \frac{1}{N} \sum_{i=1}^{N} \left( acc_2(r_i) - acc_1(r_i) \right)$$

## 📝 References

- Bjøntegaard, G. (2001). "Calculation of average PSNR differences between RD curves"
- VCEG Document: "BD-Rate/PSNR Calculus"
- IETF NETVC: Standard codec comparison methodology

## 🤝 Contributing

Để improve pipeline:
1. Test trên dữ liệu thực tế (real compression results)
2. Tune interpolation method (PCHIP vs cubic spline vs polynomial)
3. Extend cho multi-metric evaluation (ví dụ combine mIoU + PSNR)

## 📧 Support

Nếu có issue, check:
1. Input CSV format (columns, dtypes)
2. Bitrate units (kbps vs Mbps)
3. Dữ liệu đã aggregate chưa (frame-wise → video-wise → dataset-wise)
