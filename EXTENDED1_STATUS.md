# Extended 1: trạng thái triển khai

## Đã chạy được

- Giao diện: tải phiếu, căn theo mẫu, xem và sửa từng crop, xác nhận, nhận dạng bằng VietOCR, sửa nội dung và tải JSON.
- Xử lý mực xanh: lọc màu xanh để loại chữ in và dòng chấm.
- Xử lý mực đen: nếu không thấy mực xanh, so với ảnh mẫu đã căn để ước lượng nét viết đen. Người dùng vẫn có thể crop tay và sửa kết quả.
- Fine-tune pilot: `pilot_finetune.py` dùng tám nhãn đã kiểm tra bằng mắt từ hai phiếu có sẵn. Nó tạo thêm tám ảnh **mực đen mô phỏng** từ chính nét bút xanh, rồi cập nhật lớp cuối của VietOCR. Trọng số nằm ở `models/pilot_finetuned.pth` trên máy phát triển.
- Luồng cho dataset mới: `prepare_dataset.py` → gán nhãn `prepared/to_label.csv` → `export_vietocr.py` chia theo người viết → `train_extended1.py` chọn trọng số theo tập valid → `evaluate_test.py` so sánh trên tập test chưa dùng khi train.

## Kết quả kiểm thử pilot

| Điều kiện | VietOCR gốc | Pilot fine-tune |
| --- | ---: | ---: |
| Hai phiếu bút xanh, 8 trường | 7/8 | 7/8 |
| Hai phiếu đổi màu mực bằng mô phỏng, 8 trường | 5/8 | 5/8 |

Đây là **kiểm thử trên chính hai phiếu dùng để train**, không phải độ chính xác trên phiếu mới. Biến thể mực đen là ảnh tạo từ nét xanh; chưa có phiếu viết bút đen thật để kiểm chứng. Pilot chứng minh luồng huấn luyện và nạp trọng số chạy được, nhưng **chưa chứng minh mô hình tốt hơn VietOCR gốc**. Bản train thử đầu cập nhật quá nhiều tham số đã giảm mạnh kết quả, nên không dùng; pilot cuối chỉ cập nhật lớp cuối và giữ nguyên kết quả đã có.

## Chạy pilot trên máy này

Mở PowerShell trong thư mục `Demo`:

```powershell
& '.\.venv\Scripts\python.exe' '.\pilot_finetune.py'
& '.\.venv\Scripts\python.exe' '.\evaluate_pilot.py'
& '.\.venv\Scripts\python.exe' '.\ocr_review_app.py' --weights '.\models\pilot_finetuned.pth' --port 7861
```

Mở <http://127.0.0.1:7861>. Cổng 7860 vẫn có thể dùng để chạy model gốc nhằm so sánh. Hai script đầu có thể mất vài chục giây trên CPU. File trọng số `.pth` bị `.gitignore` bỏ qua vì lớn hơn giới hạn Git thường; cần chia sẻ riêng khi nhóm muốn dùng pilot trên máy khác.

## Khi có dataset mới

1. Thu phiếu của **nhiều người viết** bằng cả bút xanh lẫn bút đen. Giữ đúng mẫu phiếu hiện tại và ghi `writer_id` ổn định.
2. Chạy `prepare_dataset.py --input-dir forms --output-dir prepared`. Mở ảnh crop kiểm tra; sửa vùng lỗi trước khi gán nhãn. Điền nhãn chính xác trong `prepared/to_label.csv`.
3. Chạy `export_vietocr.py --data-dir prepared`. Script chia train/valid/test theo `writer_id`, tránh để phiếu của cùng người ở nhiều tập.
4. Chạy `train_extended1.py --data-dir prepared --output models/extended1_finetuned.pth`. Giữ tập test nguyên vẹn trong lúc chọn mô hình.
5. Chạy `evaluate_test.py --data-dir prepared --weights models/extended1_finetuned.pth` để so sánh model gốc và model mới trên **cùng tập test**. Mở giao diện với `--weights models/extended1_finetuned.pth`.

Chỉ sau bước 5 mới có thể báo cáo độ chính xác có ý nghĩa cho Extended 1. Nếu phiếu mới khác bố cục hoặc chụp thiếu vùng tiêu đề, cần kiểm tra lại phần căn/crop trước khi đánh giá OCR.
