# Phản hồi rút gọn 1 trang (gửi thầy)

## 1) Cập nhật nhanh kết quả chính
- Em đã train/evaluate lại segmentation, kết quả PIDNet trên val hiện là mean IoU = 0.8498 (500 mẫu), cao hơn rõ rệt so với run lỗi trước đó.
- Việc mIoU 15.92% trước đây là do run benchmark nhẹ/lỗi, không dùng làm kết luận chính nữa.
- Các bảng có số trùng giữa GĐ2 (23/32) và GĐ3 (25/32) là lỗi tổng hợp số liệu; em đã đối chiếu lại artifact thật.

## 2) Trả lời các comment kỹ thuật của thầy

### (A) RD/BD-rate/BD-PSNR
- Hiện em đã có RD curves và grid-search CRF nhiều điểm.
- Tuy nhiên pipeline hiện chưa có script BD-rate/BD-PSNR chính thức.
- Em sẽ bổ sung BD-rate/BD-PSNR trên cùng tập điểm RD để báo cáo đúng chuẩn codec.

### (B) Baseline lấy trung bình 2 CRF
- Baseline hiện tại được đặt theo heuristic: CRF_trad = round((CRF_ROI + CRF_NON)/2).
- Đây là anchor thực nghiệm, chưa phải chứng minh chuẩn RD.
- Em sẽ ghi rõ đây là giả định nội bộ, không khẳng định là mô tả nguyên văn từ paper.

### (C) Công thức SA-PSNR/SA-SSIM
- Công thức đã dùng nhất quán trong các script metrics/optimization/benchmark.
- Về logic, ROI đang được ưu tiên trọng số, nhưng tên biến hiện tại dễ gây hiểu nhầm.
- Em sẽ chuẩn hóa ký hiệu trọng số và đối chiếu lại ký hiệu theo paper để tránh sai diễn giải.

### (D) CRF động theo frame vs CRF cố định
- Kết quả hiện tại mới đủ cho kết luận fixed-CRF (grid-search).
- Nhánh dynamic-CRF em đang thử nhưng chưa có bằng chứng ổn định tốt hơn.
- Em sẽ làm A/B test cùng bitrate budget, rồi kết luận bằng RD + BD-rate/BD-PSNR.

### (E) Realtime 1.25 FPS
- 1.25 FPS là khi chạy full pipeline trên cùng một máy (segmentation + encode/decode + hiển thị nhiều nhánh).
- Em sẽ đổi cách gọi thành near-realtime prototype, và bổ sung benchmark khi tách vai trò xử lý hoặc tắt nhánh traditional.

### (F) Bitrate SAC và tương thích decoder
- Bitrate SAC hiện tính theo tổng hai stream (ROI + non-ROI), không phải chỉ file blend.
- Thiết bị nhận cần decode 2 stream rồi ghép lại.
- Mỗi stream vẫn là H.265, nhưng toàn hệ thống là dual-stream nên không tương đương luồng H.265 đơn truyền thống.

### (G) VCM/task-level sau nén
- Hiện có mIoU task-level cho nhánh FCM (oracle vs codec), kèm bitrate/latency.
- Chưa có artifact mAP detection hoặc lane detection sau nén trong bộ kết quả hiện tại.
- Câu hỏi “SA-PSNR tăng nhưng mAP giảm” hiện chưa đủ dữ liệu để kết luận; em sẽ bổ sung thí nghiệm task-level tương ứng.

## 3) Trạng thái “Cải tiến 1”
- Trạng thái chính thức hiện tại: tối ưu tham số cố định (fixed CRF) bằng grid-search.
- Dynamic CRF theo frame là nhánh đang nghiên cứu, chưa đưa vào kết luận chính.

## 4) Cam kết bản cập nhật tiếp theo
- Bổ sung BD-rate/BD-PSNR chuẩn hóa.
- Tách rõ kết luận fixed-CRF (đã xác nhận) và dynamic-CRF (đang thử nghiệm).
- Sửa lại toàn bộ bảng/figure theo artifact mới nhất, tránh lỗi trùng/sai số.
- Bổ sung ít nhất một bài toán task-level sau nén (detection hoặc lane) để tăng độ thuyết phục theo hướng VCM.