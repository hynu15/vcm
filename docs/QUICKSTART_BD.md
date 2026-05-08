# ⚡ QUICKSTART: BD-Rate / BD-Accuracy Pipeline

## Install & Run (30 seconds)

```bash
# 1. Activate conda environment
conda activate sac

# 2. Install tabulate (one-time only)
pip install tabulate

# 3. Run demo
cd /home/huy/sac_project/scripts
python bd_demo.py
```

Output: `outputs/bd_demo/` — xem `rd_curve.png`, `tables/bd_results_table.csv`

## Use on Real Data

Prepare CSV files với columns: `qp, bitrate_kbps, accuracy_mean`

```bash
python compute_bd_pipeline.py \
    --baseline-csv baseline_qp_data.csv \
    --propose-csv propose_qp_data.csv \
    --metric miou \
    --output-dir my_results/
```

Results: `my_results/bd_metrics.json`, `my_results/rd_curve.png`, `my_results/tables/`

## Input CSV Format

```csv
qp,bitrate_kbps,accuracy_mean
22,3000,0.835
27,2000,0.828
32,1200,0.815
37,800,0.795
```

## Output Metrics

- **BD-Rate (%)**: Bitrate savings ở cùng accuracy
  - Âm = tốt hơn ✅
  
- **BD-Accuracy**: Accuracy gain ở cùng bitrate (tuyệt đối)
  - Dương = tốt hơn ✅

## Example Results

```
BD-Rate:      -49.87%   (tiết kiệm 49.87% bitrate)
BD-Accuracy:  +0.0173   (cộng thêm 0.0173 mIoU)
```

## File Reference

- [Complete Documentation](BD_PIPELINE_README.md)
- [Demo Script](../scripts/bd_demo.py) — test trực tiếp
- [CLI Wrapper](../scripts/compute_bd_pipeline.py) — interface chính
- [Modules](../scripts/):
  - `compute_rd_points.py` — normalize & validate
  - `compute_bd.py` — tính Bjøntegaard delta
  - `plot_rd.py` — vẽ curves
  - `generate_bd_tables.py` — xuất bảng

## Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'tabulate'` | `pip install tabulate` |
| `Rate ranges do not overlap` | QP set {22,27,32,37} không đủ → thêm QP khác |
| CSV has wrong columns | Check columns: `qp, bitrate_kbps, accuracy_mean` |
| Bitrate values look wrong | Unit check: kbps không phải Mbps |

---

➡️ **Ready to evaluate?** Start with: `python bd_demo.py`
