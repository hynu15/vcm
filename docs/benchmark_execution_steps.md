# Quy Trình Thực Hiện

Tài liệu này ghi lại các bước đã làm để hoàn thiện báo cáo và tạo bộ số liệu so sánh.

## 1. Kiểm tra artifact hiện có
- Đọc `README.md` và `scripts/OPTIMIZATION_README.md`.
- Kiểm tra `outputs/metrics_comparison.csv`.
- Kiểm tra `outputs/model_benchmark/pidnet_s/` và `outputs/optimization/20260404_154235/metrics/`.

## 2. Bổ sung khả năng xuất kết quả theo tên run
- Thêm tham số `--run-name` cho `scripts/run_pidnet_benchmark.py`.
- Thêm tham số `--run-name` cho `scripts/run_resnet101_benchmark.py`.
- Sửa `scripts/model_benchmark_common.py` để lưu kết quả vào `outputs/model_benchmark/<run-name>/` khi có `run-name`.

## 3. Chạy benchmark PIDNet riêng
- Lệnh đã chạy:
  - `python scripts/run_pidnet_benchmark.py --run-name compare_pidnet_small --max-samples 20 --eval-frames 10 --latency-frames 4`
- Kết quả sinh ra:
  - `outputs/model_benchmark/compare_pidnet_small/training_pidnet_s.csv`
  - `outputs/model_benchmark/compare_pidnet_small/training_pidnet_s_summary.json`
  - `outputs/model_benchmark/compare_pidnet_small/metrics_pidnet_s_summary.csv`
  - `outputs/model_benchmark/compare_pidnet_small/latency_pidnet_s.csv`

## 4. Chạy benchmark ResNet101 riêng
- Lệnh đã chạy:
  - `python scripts/run_resnet101_benchmark.py --run-name compare_resnet101_small --max-samples 20 --eval-frames 10 --latency-frames 4`
- Kết quả sinh ra:
  - `outputs/model_benchmark/compare_resnet101_small/training_resnet101.csv`
  - `outputs/model_benchmark/compare_resnet101_small/training_resnet101_summary.json`
  - `outputs/model_benchmark/compare_resnet101_small/metrics_resnet101_summary.csv`
  - `outputs/model_benchmark/compare_resnet101_small/latency_resnet101.csv`

## 5. Trích số liệu để điền báo cáo
- SAC optimization:
  - `outputs/optimization/20260404_154235/metrics/crf_optimization_summary.csv`
  - `outputs/optimization/20260404_154235/metrics/best_config_report.txt`
- Model comparison:
  - `outputs/model_benchmark/compare_pidnet_small/*.csv`
  - `outputs/model_benchmark/compare_resnet101_small/*.csv`
- Traditional vs SAC summary:
  - `outputs/metrics_comparison.csv`

## 6. File báo cáo LaTeX đã tạo
- `report_completed.tex`
- File này đã điền lại:
  - bảng huấn luyện PIDNet,
  - bảng so sánh PIDNet vs ResNet101,
  - bảng SAC vs Traditional,
  - bảng grid search CRF,
  - bảng latency realtime,
  - phần phân tích và kết luận.

## 7. Ghi chú quan trọng
- Bộ benchmark so sánh model hiện là run nhanh trên 20 samples để có kết quả trong thời gian hợp lý.
- Kết quả latency cho thấy ResNet101 chậm hơn rõ rệt ở bước inference GPU.
- Nếu muốn số liệu chặt hơn cho luận văn, nên chạy lại benchmark trên số sample lớn hơn.
