# Thu dữ liệu và kiểm tra nhãn cho Extended 1

Bộ 50 ảnh mới đã được xử lý riêng trong `forms/dataset_20261102/` trên máy phát triển. Thư mục ảnh và nhãn được `.gitignore` loại khỏi GitHub. Xem `EXTENDED1_STATUS.md` ở thư mục gốc để biết kết quả hiện tại. Hướng dẫn dưới đây dành cho đợt thu thêm phiếu có mã người viết rõ ràng.

## Việc cần làm ngay

Thu **10 phiếu thử từ 5 người viết**, mỗi người viết 2 phiếu theo nội dung trong `pilot_assignments.csv`. Đây là đợt kiểm tra quy trình; chưa phải toàn bộ tập huấn luyện.

### Quy tắc thu phiếu

1. Dùng đúng mẫu `Demo/phieu_mau.png` hoặc bản in từ tài liệu mẫu hiện có. Không thay đổi font, vị trí hay kích thước các trường.
2. Dùng cả **bút xanh và bút đen**, ghi lại màu mực của từng phiếu để kiểm tra riêng hai nhóm.
3. Người viết chép đúng nội dung được phân công; không cần dùng thông tin cá nhân thật.
4. Chụp trọn phiếu, còn nhìn thấy tiêu đề và các nhãn in. Ảnh rõ nét, không bị che, không cháy sáng và không cắt mất mép phiếu.
5. Đặt tên ảnh đúng mẫu `w01_f01.jpg`, `w01_f02.jpg`, ..., `w05_f02.jpg`. `w01` là mã người viết, `f01` là số phiếu. Không dùng cùng một mã cho hai người khác nhau.
6. Mỗi phiếu chỉ giữ một ảnh tốt nhất trong thư mục này. Có thể lưu PNG, JPG hoặc JPEG.

## Xử lý 10 phiếu pilot

Chạy từ thư mục gốc dự án:

```powershell
python Demo/prepare_dataset.py `
  --input-dir Demo/forms `
  --output-dir Demo/dataset_v1 `
  --ground-truth Demo/forms/pilot_assignments.csv
```

Sau đó kiểm tra với **từng phiếu**:

- `Demo/dataset_v1/<tên_phiếu>/debug_fields.png`: bốn khung có trúng đúng trường không.
- Bốn crop `ho_ten.png`, `ngay_sinh.png`, `dan_toc.png`, `ton_giao.png`: có đủ nét chữ, không lẫn trường khác và không mất dấu không.
- `Demo/dataset_v1/to_label.csv`: `writer_id` và `label` được tự điền từ `pilot_assignments.csv`; vẫn phải kiểm tra bằng mắt để chắc chắn người viết đã chép đúng.

Nếu 40 crop và nhãn đều đúng, xuất tập dữ liệu VietOCR:

```powershell
python Demo/export_vietocr.py --data-dir Demo/dataset_v1
```

Nếu cả 10 phiếu đều xử lý ổn thì mới mở rộng thu dữ liệu. Mục tiêu thực tế ban đầu là ít nhất **20 người viết**, mỗi người khoảng **5 phiếu**, tương đương 100 phiếu và 400 crop. Nếu còn thời gian, tăng số người viết và số nội dung đa dạng sẽ tốt hơn tăng nhiều phiếu của cùng một người.

## Tiêu chí đạt pilot

- 10/10 ảnh đọc được và có đúng mã người viết.
- Phần lớn phiếu căn đúng; crop lỗi phải được sửa hoặc loại trước khi train.
- 40 crop có nhãn chính xác, không chứa tab hoặc xuống dòng.
- Có thể chạy `export_vietocr.py` và tạo ba tập train/valid/test theo người viết.

Không dùng tập test để chỉnh tọa độ crop, chọn checkpoint hoặc sửa tham số mô hình. Sau khi đã chia tập chính thức, giữ nguyên tập test để so sánh VietOCR gốc với VietOCR fine-tune.
