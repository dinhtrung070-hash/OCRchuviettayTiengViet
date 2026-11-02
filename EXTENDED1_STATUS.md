# Extended 1 · VietOCR fine tune trên 50 phiếu

## Phạm vi đã triển khai

- Một bố cục phiếu cố định, bốn trường: họ tên, ngày sinh, dân tộc, tôn giáo.
- Căn phiếu bằng SIFT/homography; crop theo mẫu; tách mực xanh hoặc trừ phần in của mẫu để lấy mực đen.
- Fine tune VietOCR `vgg_transformer`: giữ nguyên CNN, cập nhật transformer. Trọng số được chọn bằng tập valid.
- Giao diện Gradio: tải ảnh → xem/sửa từng crop → xác nhận crop → OCR → sửa nội dung → kiểm tra cơ bản từng trường → xuất JSON.

## Dữ liệu và giới hạn nhãn

`Dataset.zip` ngày 02/11/2026 chứa 50 ảnh JPG, không kèm đáp án hay mã người viết. Cả 50 ảnh đều căn được; chương trình tạo 200 crop. Nhãn được đọc trực tiếp từ ảnh và chuẩn hóa ngày theo `dd/mm/yyyy`. Lượt gán nhãn đầu có **45 phiếu / 180 crop**: 23 phiếu bút xanh và 22 phiếu bút đen. Năm phiếu `14, 35, 42, 43, 44` được giữ ngoài train và đánh giá vì một số chữ hoặc ngày chưa đủ chắc để gán nhãn.

Tập được chia theo **phiếu**, không phải theo người viết vì ZIP không có `writer_id`: train 31 phiếu / 124 crop, valid 7 phiếu / 28 crop, test 7 phiếu / 28 crop. Các phiếu test là `4, 5, 13, 26, 34, 36, 50`. Vì không biết liệu một người có viết nhiều phiếu hay không, **chưa thể gọi đây là đánh giá trên người viết mới**. Nhãn cũng cần nhóm kiểm tra lại với ảnh gốc trước khi dùng để báo cáo chính thức. Bước tiền xử lý được chỉnh khi đã nhìn toàn bộ bộ ảnh; tập test chỉ được giữ ngoài lựa chọn checkpoint và learning rate, chưa phải tập kiểm định độc lập hoàn toàn.

File cục bộ, bị `.gitignore` loại khỏi GitHub:

- `forms/dataset_20261102/ground_truth_first_pass.csv`: nhãn lượt đầu; `form_001`... là mã phiếu tạm, **không phải mã người viết**.
- `prepared/dataset_20261102/review.html`: đối chiếu ảnh gốc, crop và nhãn từng phiếu.
- `prepared/dataset_20261102/to_label.csv`, `split_manifest.json`, các file annotation và crop.
- `models/extended1_finetuned.pth`: checkpoint được chọn, khoảng 145 MB.

## Kết quả thực chạy

Độ đúng là số crop khớp **toàn bộ chuỗi** sau khi bỏ khoảng trắng thừa. CER là số thao tác sửa ký tự chia cho tổng ký tự nhãn; vẫn tính khác biệt dấu tiếng Việt và chữ hoa/thường.

| Mô hình | Valid: đúng hoàn toàn | Valid: CER |
| --- | ---: | ---: |
| VietOCR gốc | 4/28 | 40,4% |
| Chỉ fine tune lớp cuối, 8 epoch, LR `5e-5` | 4/28 | 37,3% |
| Fine tune transformer, 8 epoch, LR `2e-6` | 5/28 | 36,4% |
| Fine tune transformer, 8 epoch, LR `1e-5` | 5/28 | 32,9% |
| Tiếp tục từ checkpoint trên, 6 epoch, LR `5e-6` **(được chọn)** | **6/28** | **31,6%** |

Sau khi chọn checkpoint, đánh giá một lần trên 7 phiếu test:

| Mô hình | Test: đúng hoàn toàn | Test: CER |
| --- | ---: | ---: |
| VietOCR gốc | 1/28 | 47,7% |
| Fine tune được chọn | **2/28** | **37,7%** |

Trên 12 crop bút đen trong test, bản fine tune đúng 2 (gốc đúng 1); CER 28,7% (gốc 49,5%). Trên 16 crop bút xanh, cả hai bản đều đúng 0; CER 44,2% (gốc 46,4%). Họ tên và dân tộc vẫn đúng 0/7 trường mỗi loại. **Fine tune có giảm lỗi ký tự, nhưng độ đúng toàn trường còn thấp**; giao diện phải giữ bước người dùng kiểm tra và sửa kết quả. Không dùng các số này để tuyên bố mô hình đã nhận dạng tốt trên người viết mới.

## Mở demo trên máy này

Trong PowerShell tại thư mục `Demo`:

```powershell
.\start_extended1.ps1
```

Hoặc chạy trực tiếp:

```powershell
& '.\.venv\Scripts\python.exe' '.\ocr_review_app.py' --weights '.\models\extended1_finetuned.pth' --port 7861
```

Mở <http://127.0.0.1:7861>. Giao diện có hai phiếu mẫu cũ và, nếu ảnh dữ liệu còn trên máy, phiếu số 1 bút đen và số 50 bút xanh. Giữ cửa sổ PowerShell mở khi dùng lệnh trên. Nút xuất JSON ghi cả dự đoán gốc, nội dung đã sửa, cách crop và kết quả kiểm tra hợp lệ cơ bản. Trọng số và dữ liệu không nằm trên GitHub vì file lớn và phiếu có thể chứa thông tin cá nhân.

## Chạy lại quy trình

```powershell
& '.\.venv\Scripts\python.exe' '.\prepare_dataset.py' --input-dir '.\forms\dataset_20261102' --output-dir '.\prepared\dataset_20261102' --ground-truth '.\forms\dataset_20261102\ground_truth_first_pass.csv'
& '.\.venv\Scripts\python.exe' '.\make_review_page.py' --data-dir '.\prepared\dataset_20261102' --source-dir '.\forms\dataset_20261102'
& '.\.venv\Scripts\python.exe' '.\export_vietocr.py' --data-dir '.\prepared\dataset_20261102' --skip-unlabeled --allow-provisional-form-groups
```

Thông số train và lịch sử valid nằm trong các JSON cùng tên checkpoint trong `models/`. Kết quả dự đoán từng crop của test nằm ở `models/extended1_test.json`. Khi có mã người viết và nhãn đã kiểm tra, thay mã phiếu tạm bằng `writer_id` thật, chia lại tập theo người viết, train lại, rồi đánh giá trên test mới chưa dùng để chỉnh pipeline.
