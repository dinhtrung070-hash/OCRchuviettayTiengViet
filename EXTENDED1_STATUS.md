# Extended 1 · làm lại từ VietOCR gốc, fine-tune từng bước nhỏ

## Tình trạng thực tế

Demo ở `http://127.0.0.1:7861/` cho tải phiếu, tự căn và crop, sửa crop tay, xác nhận từng trường, OCR, sửa chữ và xuất JSON. Giao diện hiện cho chọn **VietOCR gốc** hoặc **fine-tune bước 1** trên cùng crop; mặc định dùng model gốc vì bản fine-tune chưa thắng trên tập test. Không nên trình bày đây là hệ thống nhận dạng chính xác cao.

## Vì sao bản trước bị lệch

Hai phiếu mẫu ban đầu là một bản in có tiêu đề chữ hoa; 50 phiếu mới dùng `phieu_mau.png` với tiêu đề chữ thường và vị trí chữ hơi khác. Bản trước dùng chung khung cắt chặt cho cả hai. Bước loại chấm còn lấy nét in từ mẫu rồi xóa cả vùng quanh nét in, làm mất một phần nét viết. Trên hai phiếu mẫu cũ, VietOCR gốc với crop cũ đúng **7/8** trường; cùng model với crop sau đó chỉ đúng **1/8**. Đây là lỗi đầu vào, không phải bằng chứng fine-tune tự nó làm model quên hết.

Đã sửa: nhận biết hai bản in bằng số điểm SIFT khớp với ảnh tham chiếu, dùng khung cắt riêng; với mực xanh giữ các thành phần nét viết đủ lớn rồi bỏ chấm nằm xa cụm chữ, **không xóa theo mẫu in**. Mực đen vẫn dùng nhánh trừ nền mẫu hiện có; phần này còn làm mất nét khi chữ đè lên dòng chấm. Nút crop tay và sửa nội dung vẫn cần thiết.

## Dữ liệu

`Dataset.zip` có 50 ảnh, không có đáp án hay mã người viết. Đã căn được 50/50 ảnh; có 200 crop. Nhãn đọc trực tiếp từ ảnh cho 45 phiếu / 180 crop (23 phiếu xanh, 22 phiếu đen); năm phiếu 14, 35, 42, 43, 44 còn mơ hồ nên để ngoài train và đánh giá. Nhãn là lượt đầu, cần người trong nhóm kiểm tra lại trên ảnh gốc.

Chia theo **mã phiếu tạm**: train 31 phiếu / 124 crop, valid 7 / 28, test 7 / 28. Chưa có ID người viết thật nên không thể tuyên bố kết quả trên người viết mới. Tập test cũng đã được xem trong lần thử trước, nên mọi con số dưới đây mang tính khảo sát; cần ảnh mới chưa dùng để chỉnh pipeline cho báo cáo cuối.

Ảnh, nhãn, crop và trọng số nằm cục bộ, bị `.gitignore` loại khỏi GitHub. Dữ liệu mới đã chuẩn bị ở `prepared/dataset_20261103_component/`; nhãn gốc ở `forms/dataset_20261102/ground_truth_first_pass.csv`.

## Cách fine-tune mới

Mỗi phép thử được so với model gốc trên cùng crop. Bước 1 bắt đầu từ `models/pretrained_vgg_transformer.pth`, khóa CNN và toàn bộ transformer trừ lớp dự đoán ký tự `transformer.fc`, learning rate `1e-5`, tối đa ba epoch. Chọn checkpoint ở epoch 2 bằng valid; epoch 3 bị loại vì CER tăng. Bước 2 thử mở transformer từ checkpoint bước 1 với learning rate `1e-6`, hai epoch; cả hai bị loại vì CER valid xấu hơn. Quy tắc lưu checkpoint: số trường đúng và CER trên valid phải cải thiện, đồng thời hai phiếu mẫu cũ không được giảm số trường đúng. **Không dùng kết quả test để chọn epoch.**

| Model / checkpoint | Valid đúng toàn trường | Valid CER | Hai mẫu cũ |
| --- | ---: | ---: | ---: |
| VietOCR gốc, crop đã sửa | 4/28 | 36,8% | 7/8 |
| Fine-tune bước 1, epoch 2 | 4/28 | 36,0% | 7/8 |
| Thử transformer bước 2, epoch 2 (loại) | 4/28 | 39,5% | 7/8 |

Đánh giá khảo sát trên 7 phiếu test với **cùng crop mới**:

| Model | Test đúng toàn trường | Test CER |
| --- | ---: | ---: |
| VietOCR gốc | 1/28 | 51,5% |
| Fine-tune bước 1 | 1/28 | 51,9% |

Vì fine-tune bước 1 hơi xấu hơn trên test, demo để model gốc làm mặc định và cho chọn bản fine-tune để đối chiếu. Mức cải thiện valid chỉ khoảng 0,9 điểm phần trăm CER; chưa có bằng chứng bản fine-tune tổng quát hóa tốt hơn. Trên phiếu 50, crop dân tộc đã bỏ được dãy chấm dài, nhưng VietOCR gốc vẫn đọc `Dây` thay cho `Tày`.

## Cách chạy demo

Mở PowerShell ở thư mục `Demo`:

```powershell
.\start_extended1.ps1
```

Mở <http://127.0.0.1:7861/>. Chọn phiếu mẫu hoặc tải ảnh, xem và xác nhận bốn crop. Chọn model trước khi bấm OCR. Có thể đổi model rồi bấm OCR lại trên cùng crop để so sánh. Nếu chữ sai, sửa ô kết quả trước khi xuất JSON. Trọng số fine-tune cục bộ: `models/incremental_component_head_step1.pth`; máy khác cần nhận file này riêng. Nếu chưa có file đó, chạy trực tiếp `ocr_review_app.py` để dùng model gốc trên cổng 7860.

## Tái hiện bước train nhỏ

Sau khi đặt ảnh và nhãn cục bộ đúng đường dẫn:

```powershell
& '.\.venv\Scripts\python.exe' '.\prepare_dataset.py' --input-dir '.\forms\dataset_20261102' --output-dir '.\prepared\dataset_20261103_component' --ground-truth '.\forms\dataset_20261102\ground_truth_first_pass.csv'
& '.\.venv\Scripts\python.exe' '.\export_vietocr.py' --data-dir '.\prepared\dataset_20261103_component' --skip-unlabeled --allow-provisional-form-groups
& '.\.venv\Scripts\python.exe' '.\prepare_dataset.py' --input-dir '.' --output-dir '.\prepared\dataset_20261103_component\pilot' --ground-truth '.\pilot_labels.csv'
& '.\.venv\Scripts\python.exe' '.\train_extended1.py' --data-dir '.\prepared\dataset_20261103_component' --regression-dir '.\prepared\dataset_20261103_component\pilot' --output '.\models\incremental_component_head_step1.pth' --scope head --epochs 3 --lr 0.00001
& '.\.venv\Scripts\python.exe' '.\evaluate_test.py' --data-dir '.\prepared\dataset_20261103_component' --weights '.\models\incremental_component_head_step1.pth' --output '.\models\incremental_component_head_step1_test.json'
```

Lịch sử train nằm trong JSON cạnh checkpoint; dự đoán từng trường của test nằm trong `models/incremental_component_head_step1_test.json`. Bước kế tiếp: hai người kiểm tra nhãn, gắn ID người viết thật, thu phiếu mới chưa xem, xử lý mực đen đè lên dòng chấm, rồi chỉ fine-tune tiếp nếu valid và tập kiểm tra mới đều tốt hơn.
