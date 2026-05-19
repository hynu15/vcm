# Điều chỉnh so sánh bitrate SAC vs. truyền thống

## Vấn đề ban đầu (mục 6.4 báo cáo)

Trong thực nghiệm cũ với `CRF_roi=25, CRF_non=32`, công thức tính CRF baseline là:

```python
crf_trad = int(round((25 + 32) / 2.0))  # = round(28.5) = 28  (Python banker's rounding → 28)
```

Kết quả:

| Phương án    | Bitrate   |
|-------------|-----------|
| SAC (25/32) | 22.30 Mbps |
| Trad (CRF=28) | 13.91 Mbps |
| Chênh lệch  | **+60.3%** |

Đây là so sánh **không công bằng**: baseline được cấp ít bit hơn SAC đến 60%, dù đây là một operating point "tương đương". Nguyên nhân công thức `round(28.5) = 28` (banker's rounding làm tròn về số chẵn) cộng với việc CRF_roi=25 cấp chất lượng cao hơn mức cần thiết.

---

## Cách bài báo gốc xử lý

Wang et al. chọn cặp CRF thực nghiệm sao cho **tổng bitrate SAC ≈ bitrate traditional**:

| Cặp SAC         | Traditional | Lý do |
|-----------------|-------------|-------|
| `roi=23, non=32` | `CRF=27`   | `(23+32)//2 = 27` — bitrate xấp xỉ nhau |
| `roi=18, non=27` | `CRF=23`   | Paper xác định thực nghiệm; `(18+28)//2=23` với gap=10 |

---

## Thay đổi code đã thực hiện

### 1. Công thức `crf_trad` — đổi từ `round` sang `//` (floor division)

**File:** `model_benchmark_common.py` (2 chỗ), `optimize_sac_crf.py` (2 chỗ), `run_segmentation_rd_pipeline.py` (1 chỗ)

```python
# Cũ — gây sai lệch do banker's rounding
crf_trad = int(round((crf_roi + crf_non) / 2.0))   # (23+32)/2=27.5 → round → 28 ✗

# Mới — khớp với paper
crf_trad = (crf_roi + crf_non) // 2                 # (23+32)//2 = 27 ✓
```

Kiểm tra tất cả các cặp quan trọng:

| roi | non | Cũ (round) | Mới (//) | Paper |
|-----|-----|-----------|---------|-------|
| 23  | 32  | 28        | **27**  | 27 ✓ |
| 25  | 32  | 28 hoặc 29| **28**  | — |
| 28  | 37  | 32 hoặc 33| **32**  | 32 ✓ |
| 33  | 42  | 38        | **37**  | 37 ✓ |
| 18  | 27  | 22        | **22**  | 23* |

> *Cặp (18,27): floor cho 22 thay vì 23 của paper. Đây là lệch nhỏ; paper xác định 23 thực nghiệm. Có thể dùng `(18,28)//2=23` nếu muốn khớp chính xác.

### 2. Default CRF_roi — đổi từ 25 → 23

**File:** `run_pidnet_benchmark.py`, `run_resnet101_benchmark.py`

```
Cũ: --crf-roi 25   →   crf_trad = round((25+32)/2) = 28,  chênh ~60%
Mới: --crf-roi 23  →   crf_trad = (23+32)//2 = 27,        bitrate ≈ nhau
```

### 3. Operating points RD pipeline — đổi gap từ 16 → 9

**File:** `run_segmentation_rd_pipeline.py`

```python
# Cũ — gap 16 giữa roi và non, rất khác paper
OperatingPoint("22", 20, 36)   # non-roi = roi + 16
OperatingPoint("27", 25, 41)
OperatingPoint("32", 30, 46)
OperatingPoint("37", 35, 51)

# Mới — gap 9, khớp với paper (23,32)
OperatingPoint("22", 18, 27)   # (18+27)//2 = 22 ✓
OperatingPoint("27", 23, 32)   # (23+32)//2 = 27 ✓  ← cặp chính của paper
OperatingPoint("32", 28, 37)   # (28+37)//2 = 32 ✓
OperatingPoint("37", 33, 42)   # (33+42)//2 = 37 ✓
```

---

## Tại sao `//` (floor) là đúng về mặt kỹ thuật?

Khi SAC encode hai stream riêng biệt:
- ROI stream (CRF thấp, chất lượng cao) → nhiều bit hơn mức "trung bình"
- Non-ROI stream (CRF cao, nén mạnh) → ít bit nhưng có overhead container

Tổng bitrate SAC = `bitrate_roi + bitrate_non` > `bitrate_full_frame_at_avg_CRF`

Vì mã hóa riêng kém hiệu quả hơn mã hóa chung (mất inter-region redundancy), **traditional cần CRF thấp hơn một chút** (= nhiều bit hơn) để bằng tổng SAC. Floor division cho CRF thấp hơn round-half-up, đúng hướng.

---

## Kết quả dự kiến sau khi chạy lại

| Cặp SAC       | Trad CRF | Bitrate SAC | Bitrate Trad | Chênh lệch |
|--------------|----------|-------------|--------------|-----------|
| roi=23, non=32 | 27     | ~19–22 Mbps | ~18–22 Mbps  | < 10%      |

*(Cần chạy lại thực nghiệm trên 20 frame để xác nhận số liệu chính xác)*

---

## Điểm cần bạn xem xét

1. **Cặp (18,27) cho QP~22**: floor cho `crf_trad=22`, trong khi paper dùng 23. Có muốn đổi thành `(18,28)` để `(18+28)//2=23` khớp paper không?

2. **`sac_compression_x265.py`**: File này đã đúng (`CRF_ROI=23, CRF_NON=32`, dùng `int(27.5)=27`) — không cần thay đổi.

3. **Bảng số liệu mục 6.4**: Sau khi chạy lại với cặp (23/32) vs CRF=27, kết quả sẽ thay đổi. Nếu bitrate xấp xỉ nhau và SA-PSNR vẫn cao hơn, kết luận sẽ công bằng và thuyết phục hơn.
