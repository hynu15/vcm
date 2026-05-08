# Phản hồi các comment của thầy (cập nhật 2026-04-28)

## 1) Phần trả lời của em (giữ nguyên)

à vâng ạ, em đang train lại data đây ạ, 
- chỉ sô mIoU hôm bữa em train bị lỗi, em train lại trên toàn bộ tập dữ liệu train cho 2 mô hình ccnet và pidnet thì có thấy được pidnet có chỉ số mIoU cao hơn 
{
  "model_name": "pidnet_s",
  "model_path": "/home/huy/sac_project/models/best_pidnet.pth",
  "num_samples": 500,
  "mean_iou": 0.849823079432895,
  "per_class_iou": {
    "roi": 0.9117353979893461,
    "sky": 0.8681730765578726,
    "construction": 0.765232798782746,
    "nature": 0.8541510444016157
}
- so với mô hình ccnet theo như của bài báo là ccnet là mIou là tầm: 0.82 trên tập dữ liệu val. ( chỉ số của 4 vùng củ thể như sau :"roi": 0.91, "sky": 0.8239; "construction": 0.775 , "nature": 0.82
--> trong bài báo thì tập xác thưcj là 0.8089
- kết quả của realtime thấp là tại em hôm bữa vừa nén, vừa giải nén với các thứ trên cùng 1 máy, kiểu nó phải xử lý quá nhiều thứ, xuất hiện cả 3 khung hình: ori, sac, với x265 nên máy nó không xử lý được, nếu tách ra thì em nhớ không nhầm là đươc 25fps -> cái này em cũng chưa làm thêm tại em đang muốn train lại dữ liệu để kết quả mIoU của mô hình được tốt hơn rồi mới áp dụng realtime sau
cái GDD2 ý hôm bữa em train bị lỗi model, em có xem lại rồi. 
7. thiết bị cần decode 2 stream riêng biệt sau đó (+) lại với nhau -> nó vẫn tương thích được với H265, em làm cũng không thấy lỗi gì, chỉ có cái là đôi khi xuất hiện các vùng dải đen giữa vùng roi và non-roi thôi ạ 
mấy cái chỉ số cùng PSNR, SSIM, SA-PSNR, SA-SSIM mà ảnh bảo em thì em cũng thấy nó sai rồi, bữa em làm em train xong em đáng lẽ phải nên kiểm tra kỹ hơn anh ạ
11. đây là tối ưu theo tham số cố định, crf động theo từng frame em cũng đang làm mà chưa thấy kết quả tốt hơn ý.

## 2) Bổ sung các ý còn thiếu theo từng comment

### Comment 1: Cần RD curves nhiều mức CRF/QP và tính BD-rate / BD-PSNR
- Hiện tại project đã có RD curve (plot bitrate vs SA-PSNR/SA-SSIM) và grid-search CRF: xem [scripts/OPTIMIZATION_README.md](scripts/OPTIMIZATION_README.md) và [outputs/optimization/20260404_154235/metrics/crf_optimization_summary.csv](outputs/optimization/20260404_154235/metrics/crf_optimization_summary.csv).
- Tuy nhiên, chưa thấy script tính BD-rate/BD-PSNR chính thức trong pipeline hiện tại.
- Kết luận phản hồi: Em mới dừng ở RD curves và objective score; em sẽ bổ sung bước tính BD-rate/BD-PSNR trên cùng tập điểm RD cho baseline và SAC để báo cáo chuẩn theo thông lệ codec.

### Comment 2: Tại sao baseline lấy trung bình hai CRF? Có đúng paper không?
- Trong code hiện tại, baseline truyền thống đang được đặt bằng CRF trung bình: crf_trad = round((CRF_ROI + CRF_NON)/2), xem [scripts/sac_compression_x265.py](scripts/sac_compression_x265.py) và [scripts/optimize_sac_crf.py](scripts/optimize_sac_crf.py).
- Đây là giả định thực nghiệm để có anchor gần mức nén trung bình, chưa phải chứng minh chặt theo chuẩn RD/BD.
- Kết luận phản hồi: Em sẽ ghi rõ đây là heuristic nội bộ (không khẳng định là mô tả nguyên văn từ paper), và sẽ chuyển sang so sánh nhiều điểm RD + BD-rate/BD-PSNR để tránh phụ thuộc một anchor duy nhất.

### Comment 3: Kiểm tra công thức SA-PSNR/SA-SSIM
- Công thức đang dùng nhất quán ở nhiều script: [scripts/calculate_metrics.py](scripts/calculate_metrics.py), [scripts/optimize_sac_crf.py](scripts/optimize_sac_crf.py), [scripts/model_benchmark_common.py](scripts/model_benchmark_common.py).
- Dạng triển khai hiện tại là:
  - r_roi = CRF_ROI/(CRF_ROI+CRF_NON), r_non = CRF_NON/(CRF_ROI+CRF_NON)
  - SA-PSNR = r_non * PSNR_ROI + r_roi * PSNR_NON
- Về bản chất, ROI đang được nhân hệ số lớn hơn (do CRF_NON > CRF_ROI). Tức logic ưu tiên ROI có tồn tại, nhưng tên biến r_roi/r_non dễ gây hiểu nhầm.
- Kết luận phản hồi: Em sẽ chuẩn hóa ký hiệu để rõ nghĩa (ví dụ w_roi, w_non) và đối chiếu lại ký hiệu đúng theo paper để tránh nhầm lẫn diễn giải.

### Comment 4: Chứng minh CRF động theo frame tốt hơn CRF cố định
- Hiện repo chưa có artifact cho CRF động theo frame; mới có tối ưu tham số cố định (grid-search).
- Kết luận phản hồi: Em chưa đủ bằng chứng để kết luận CRF động tốt hơn. Em sẽ làm A/B test cùng tập frame, cùng preset, cùng bitrate budget; sau đó báo cáo bằng RD + BD-rate/BD-PSNR và kiểm định ý nghĩa thống kê.

### Comment 5: mIoU 15.92% quá thấp
- Điểm này đã được cập nhật bằng kết quả eval mới: mean_iou = 0.849823 trên 500 mẫu val, xem [outputs/metrics/segmentation_val_summary_pidnet_s_best_pidnet.json](outputs/metrics/segmentation_val_summary_pidnet_s_best_pidnet.json).
- Kết luận phản hồi: Kết quả 15.92% là từ run lỗi/benchmark nhẹ trước đó; hiện đã có kết quả val mới cao hơn và phù hợp hơn để dùng làm căn cứ cho ROI mask.

### Comment 6: Vì sao gọi realtime khi chỉ 1.25 FPS?
- Số ~1.25 FPS xuất phát từ pipeline demo chạy đồng thời nhiều nhánh (segmentation + encode/decode SAC + traditional + hiển thị 3 panel), xem [scripts/realtime_sac_x265_demo.py](scripts/realtime_sac_x265_demo.py) và [scripts/realtime_sac_x265_lowlatency.py](scripts/realtime_sac_x265_lowlatency.py).
- Kết luận phản hồi: Mức 1.25 FPS là realtime prototype trên single-machine full pipeline, chưa phải cấu hình deploy tách vai trò. Em sẽ đổi cách diễn đạt thành near-realtime/prototype và bổ sung benchmark khi tách encode/decode hoặc disable nhánh traditional.

### Comment 7: Bitrate SAC là tổng 2 stream hay video blend? Thiết bị nhận decode mấy stream?
- Trong code đánh giá, bitrate SAC đang tính bằng tổng bitrate roi.mp4 + nonroi.mp4, không phải bitrate của file blend cuối, xem [scripts/optimize_sac_crf.py](scripts/optimize_sac_crf.py) và [scripts/model_benchmark_common.py](scripts/model_benchmark_common.py).
- Thiết bị nhận cần decode 2 stream riêng rồi ghép theo mask.
- Kết luận phản hồi: Vẫn dùng chuẩn H.265 cho từng stream, nhưng kiến trúc hệ thống là dual-stream nên không tương đương một luồng H.265 đơn với decoder truyền thống thuần túy.

### Comment 8: Hướng VCM chưa có mAP/mIoU/lane detection sau nén?
- Với nhánh FCM hiện có mIoU task-level (oracle vs codec) và bitrate/latency trong pipeline feature coding, xem [fcm_simple_project/scripts/04_metrics/evaluate_feature_pipeline.py](fcm_simple_project/scripts/04_metrics/evaluate_feature_pipeline.py) và [fcm_simple_project/docs/pipeline_notes_vi.md](fcm_simple_project/docs/pipeline_notes_vi.md).
- Chưa có artifact cho mAP detection hoặc lane-detection sau nén trong repo hiện tại.
- Câu hỏi "SA-PSNR tăng nhưng mAP giảm" hiện chưa có đủ dữ liệu để kết luận.
- Kết luận phản hồi: Em sẽ bổ sung đánh giá task-level sau nén (ít nhất detection hoặc lane) để chứng minh quality metric và task metric không mâu thuẫn.

### Comment 9: Điều chỉnh CRF_ROI (và CRF_NON) theo mật độ ROI từng frame
- Hiện code SAC đang dùng CRF cố định theo run; chưa thấy cơ chế adaptive theo ROI density trong artifact chính.
- Kết luận phản hồi: Đây là hướng đang làm, nhưng chưa có kết quả tốt hơn cố định nên em chưa đưa vào kết luận chính.

### Comment 10: GĐ2 (23/32) và GĐ3 (25/32) ở Bảng 6.10 bị trùng chỉ số
- Em đã kiểm tra và xác nhận bảng cũ có lỗi nhập số.
- Dữ liệu thật trong summary CSV cho 23/32 và 25/32 khác nhau, xem [outputs/optimization/20260404_154235/metrics/crf_optimization_summary.csv](outputs/optimization/20260404_154235/metrics/crf_optimization_summary.csv).
- Kết luận phản hồi: Em sẽ thay số liệu trong báo cáo theo file summary để tránh trùng nhầm.

### Comment 11: [CẢI TIẾN 1] là tối ưu cố định hay CRF động?
- Theo artifact hiện tại: đây là tối ưu tham số cố định bằng grid-search.
- CRF động theo từng frame là nhánh đang thử nghiệm, chưa đủ kết quả ổn định để đưa vào kết luận chính.

## 3) Cam kết chỉnh báo cáo lần tới
- Bổ sung BD-rate/BD-PSNR chuẩn hóa từ bộ điểm RD.
- Tách rõ hai mức kết luận: (A) fixed-CRF đã xác nhận, (B) dynamic-CRF còn đang thử nghiệm.
- Sửa toàn bộ bảng/figure theo số liệu artifact mới nhất để tránh trùng hoặc sai copy.
- Bổ sung task-level metric sau nén cho hướng VCM (ít nhất 1 tác vụ ngoài segmentation).