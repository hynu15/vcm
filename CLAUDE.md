# CLAUDE.md

File này cung cấp hướng dẫn cho Claude Code (claude.ai/code) khi làm việc với code trong repository này.

## Tổng quan dự án

Code nghiên cứu Semantic-Aware Compression (SAC) nén video Cityscapes bằng cách encode vùng ROI và non-ROI với CRF x265 khác nhau, sau đó so sánh với baseline truyền thống (single-stream). Các artifact nặng (dataset, checkpoint, video) nằm ở `data/`, `models/`, `outputs/` và bị git-ignore — chỉ có `scripts/` là source code.

## Môi trường

- Python: dự án dùng `./.venv/` (Python 3.10). Activate bằng `source .venv/bin/activate` hoặc gọi trực tiếp `./.venv/bin/python ...`.
- Công cụ ngoài: `ffmpeg` với `libx265` là bắt buộc cho mọi script encoding. Hàm helper `run_ffmpeg(...)` trong hầu hết các script gọi shell ra `ffmpeg` và raise lỗi khi exit code khác 0.
- GPU: code tự phát hiện CUDA (`torch.device('cuda' if available else 'cpu')`). Benchmark latency chạy cả GPU lẫn CPU khi có CUDA.

## Chạy scripts

Các script trong `scripts/` import lẫn nhau theo kiểu **sibling module** (ví dụ `from train_segmentation import load_segmentation_model`). Chúng không phải là package — cần chạy từ thư mục `scripts/` để import hoạt động:

```bash
cd /home/huy/sac_project/scripts
../.venv/bin/python <script>.py [args]
```

Nhiều script xác định đường dẫn bằng `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, nên thư mục làm việc không ảnh hưởng đến I/O — chỉ ảnh hưởng đến việc resolve sibling import.

## Các lệnh thường dùng

Train / benchmark model từ đầu đến cuối (training → SAC quality metrics → latency):

```bash
# PIDNet (nhẹ, mặc định)
../.venv/bin/python run_pidnet_benchmark.py --run-name pidnet_demo --max-samples 20 --eval-frames 10 --latency-frames 4

# ResNet101 + CrissCross Attention (nặng)
../.venv/bin/python run_resnet101_benchmark.py --run-name resnet101_demo --max-samples 20 --eval-frames 10 --latency-frames 4
```

Kết quả ghi vào `outputs/model_benchmark/<run-name>/{training_*,metrics_*,latency_*}.csv` và checkpoint `best_<model>.pth`. Nếu không dùng `--run-name`, kết quả lưu vào `outputs/model_benchmark/<model_name>/` và sẽ ghi đè lần chạy trước.

Grid-search CRF (tối ưu RD trên Cityscapes val):

```bash
../.venv/bin/python optimize_sac_crf.py --num-frames 20 --roi-crf-list 20,23,25 --non-crf-list 30,32,35 --objective-lambda 0.4
../.venv/bin/python visualize_optimization_results.py outputs/optimization/<TIMESTAMP>/metrics/crf_optimization_summary.csv
```

Hàm mục tiêu là `score = Δ SA-PSNR_vs_Trad - λ × Δ Bitrate`. λ thấp ưu tiên chất lượng; λ cao ưu tiên bitrate. Xem `scripts/OPTIMIZATION_README.md` để biết các recipe quét đầy đủ.

Pipeline RD đầy đủ tại nhiều operating point (ghi RD curves + BD metrics dùng PCHIP):

```bash
# Dual-stream (SAC vs traditional)
../.venv/bin/python run_segmentation_rd_pipeline.py --num-frames 20 --preset slow

# Biến thể single-stream (làm mờ non-ROI thay vì split-encode)
../.venv/bin/python run_segmentation_rd_pipeline_single.py --num-frames 20 --preset slow
```

Chạy SAC với video tùy chỉnh hoặc webcam live:

```bash
../.venv/bin/python test_video_sac.py --video /path/to/clip.mp4 --num-frames 20 [--force-sky-nonroi]
../.venv/bin/python realtime_sac_x265_demo.py --camera 0 --crf-roi 23 --crf-non 32
../.venv/bin/python realtime_sac_x265_lowlatency.py --camera 0
```

Chuẩn bị nhãn 4-class từ Cityscapes gtFine (một lần duy nhất, trước khi train):

```bash
../.venv/bin/python prepare_4class_labels.py
```

Repo này không có test runner, linter, hay build system nào được cấu hình.

## Kiến trúc

### Cấu trúc dữ liệu (bị git-ignore)

```
data/gt_4class/
  leftImg8bit_trainvaltest/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png   # frame đầu vào
  {train,val}/<city>/*_gtFine_4class.png                                       # nhãn 4-class
models/best_pidnet.pth    # checkpoint mặc định (PIDNet)
models/best_ccnet.pth     # checkpoint legacy (ResNet101+CCNet); fallback trong một số script
outputs/                  # toàn bộ artifact được sinh ra: optimization, benchmark, RD pipeline
```

Class ID trong nhãn 4-class: `0=ROI` (đường/xe/người — mọi thứ không thuộc 3 class dưới), `1=sky`, `2=construction`, `3=nature`. Cách mapping từ `_labelIds.png` Cityscapes sang ID này nằm trong `prepare_4class_labels.py`.

### Các model segmentation (`scripts/train_segmentation.py`)

File này vừa là module có thể import vừa chứa toàn bộ định nghĩa model:

- `PIDNetSegmentor` — mạng ba nhánh P/I/D nhẹ, là model mặc định. Lưu thành `best_pidnet.pth`.
- `LegacyCCNetResNet101` — backbone ResNet101 với hai block CrissCross Attention. Lưu thành `best_ccnet.pth`. Class cũng được re-export là `CCNet` để tương thích ngược.
- `build_segmentation_model(model_name, ...)` — factory. Nhận `'pidnet_s'`, `'ccnet'`, hoặc `'resnet101'`.
- `load_segmentation_model(path, device, ...)` — loader checkpoint. Đọc `meta.model_name` từ dict checkpoint; fallback sang `infer_model_name_from_state_dict` (có key `backbone./cc1./cc2.` ⇒ CCNet, ngược lại ⇒ PIDNet). Hầu hết script dùng loader này.
- Dataset `Cityscapes4Class` — ghép cặp `_leftImg8bit.png` với `_gtFine_4class.png` theo từng city.

### Pipeline nén SAC

Hình dạng pipeline chung trong `sac_compression_x265.py`, `optimize_sac_crf.py`, `test_video_sac.py`, `model_benchmark_common.py::run_sac_metrics`, và các script RD pipeline:

1. Chạy model segmentation trên từng frame ở 512×1024 để lấy dự đoán class.
2. Xây dựng ROI mask nhị phân (`mask == 0`) và đưa qua `macroblock_align_filter(mask, 16)` để biên ROI snap vào macroblock 16 pixel (nếu không x265 sẽ rò tín hiệu ROI/non-ROI qua block boundaries).
3. Tách frame: `roi_img = bitwise_and(orig, roi)`, `non_img = bitwise_and(orig, ~roi)`. Ghi mỗi cái ra chuỗi PNG tạm.
4. Encode từng stream với `ffmpeg -c:v libx265 -x265-params aq-mode=0 -crf <crf>` — `aq-mode=0` là quan trọng; SAC kiểm soát chất lượng theo vùng qua việc tách stream, không dùng adaptive quant của x265.
5. Merge ROI và non-ROI lại bằng ffmpeg `blend=all_mode=addition` (hoạt động vì hai stream có vùng non-zero không chồng lên nhau).
6. Encode baseline single-stream tại `crf_trad = round((crf_roi + crf_non)/2)` để so sánh.
7. Tính PSNR / SSIM / SA-PSNR / SA-SSIM và tỷ lệ bitrate. Trọng số SA-PSNR là `r_non * psnr_roi + r_roi * psnr_non` với `r_roi = crf_roi / (crf_roi + crf_non)` — lưu ý cross-weighting (đặt trọng số cao hơn vào metric của vùng CRF *thấp hơn*).

`encode_propose_v2.py` là phương án thực nghiệm riêng: thay vì tách frame, nó xây dựng x265 `qpfile` từ thành phần class từng frame (`get_qp_delta` theo class) — hữu ích khi so sánh các biến thể SAC single-stream.

### Benchmark runner (`scripts/model_benchmark_common.py`)

`run_full_benchmark(args)` là entry point chung dùng bởi `run_pidnet_benchmark.py` và `run_resnet101_benchmark.py`. Nó nối tiếp:

- `train_and_log` — Adam + AMP, BCE+Dice loss, lưu checkpoint của epoch có mIoU tốt nhất.
- `run_sac_metrics` — chạy pipeline SAC trên `--eval-frames` frame và ghi CSV per-frame + summary.
- `run_latency_benchmark` — đo thời gian từng giai đoạn (`preprocessing`, `inference`, `mask_split`, `x265_encode_decode`, `blend`, `total`) trên cả GPU lẫn CPU, báo cáo mean ms + FPS.

Lưu ý: `macroblock_align_filter` bị duplicate ở nhiều script (các cài đặt khác nhau tồn tại — `model_benchmark_common.py` thậm chí định nghĩa nó hai lần, lần sau ghi đè lần trước). Khi sửa thì cần cập nhật tất cả bản sao.

### Pipeline RD / BD

`run_segmentation_rd_pipeline.py` và `run_segmentation_rd_pipeline_single.py` encode tại bốn operating point QP, decode, chạy PIDNet trên frame đã decode, tính mIoU so với ground truth, rồi xây dựng RD curve và BD metrics trực tiếp bằng `scipy.interpolate.PchipInterpolator`. Tài liệu đi kèm trong `docs/BD_PIPELINE_README.md` và `docs/QUICKSTART_BD.md` có đề cập các script độc lập (`compute_bd_pipeline.py`, `bd_demo.py`, v.v.) — các script đó **không tồn tại** trong repo này; phần tính BD được inline trong hai script pipeline trên.

## Lưu ý khi chỉnh sửa

- Script trộn lẫn tiếng Anh và tiếng Việt trong comment và print statement; không dịch khi không được yêu cầu.
- File `safe_modify.py` và `test_loop.py` ở root repo là script nháp, không thuộc pipeline.
- `scripts/calculate_metrics.py` hiện ở trạng thái deleted trong `git status` — không tái tạo lại; phần tính metric hiện tại nằm trong `model_benchmark_common.py` và các script RD pipeline.
- Model segmentation mặc định là PIDNet. `sac_compression_x265.py` là ngoại lệ duy nhất hardcode `best_ccnet.pth` — hầu hết script còn lại ưu tiên `best_pidnet.pth` và chỉ fallback sang CCNet khi file đó không tồn tại.
