# OCR phiếu viết tay tiếng Việt · Extended 1

Mẫu có bốn trường: họ tên, ngày sinh, dân tộc, tôn giáo. Pipeline nhận biết hai bản in hiện có, căn phiếu bằng SIFT + homography, cắt bốn vùng và lọc chấm bảo toàn nét viết cho cả mực xanh/đen. Fine-tune VietOCR và số liệu thực chạy ở [EXTENDED1_STATUS.md](EXTENDED1_STATUS.md). Kết quả vẫn cần người dùng kiểm tra và sửa.

## Cài trên máy mới sau khi clone

Dùng Python 3.12 và PowerShell. Mở terminal **ngay trong thư mục repo** (thư mục chứa file README này):

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-demo.txt
```

Đây là môi trường CPU đã dùng để thử giao diện. Nếu muốn chạy GPU, chọn lệnh cài PyTorch phù hợp từ trang chính thức của PyTorch.

## Chạy giao diện demo

Trong thư mục repo, chạy:

```powershell
& '.\.venv\Scripts\python.exe' '.\ocr_review_app.py'
```

Mở <http://127.0.0.1:7860>. Hai phiếu `phieu_da_viet.png` và `phieudaviet1.png` có sẵn ngay trong giao diện để thử. Quy trình:

1. Tải ảnh hoặc chọn một phiếu mẫu, bấm **Căn phiếu và cắt tự động**.
2. Xem bốn ảnh crop. Bật/tắt **Lọc dòng chấm** để so với ảnh gốc. Nếu vùng nào sai, bấm **Sửa crop** ở trường đó, dùng công cụ **Crop**, bấm **Apply**, rồi **Dùng crop vừa chọn**. Nếu căn tự động thất bại, sửa crop tay cả bốn trường.
3. Tích **Crop đúng** cho đủ bốn trường rồi bấm **Xác nhận crop và chạy VietOCR**.
4. Xem các cảnh báo, đối chiếu ảnh và sửa nội dung, sau đó bấm **Xác nhận nội dung và tải JSON**. JSON giữ OCR thô, đề xuất nhận dạng theo trường và nội dung đã sửa. Đề xuất ngày/danh mục cũng có thể sai.

Demo mặc định dùng **VietOCR vgg_transformer pretrained**, chưa phải mô hình đã fine-tune. Repo không chứa file trọng số 145 MB; nếu chưa có `models/pretrained_vgg_transformer.pth`, VietOCR tải trọng số gốc từ URL trong `models/vgg_transformer.yml` ở lần OCR đầu tiên. Cần Internet cho lần tải này. Khi đã có trọng số fine-tune, chạy `ocr_review_app.py --weights đường_dẫn_đến_file.pth` để dùng trọng số mới. Toàn bộ demo chỉ phục vụ trên máy này (`127.0.0.1`), chạy bằng CPU nên lần OCR đầu có thể chậm.

Trên máy đã train Extended 1, chạy `.\start_extended1.ps1` rồi mở <http://127.0.0.1:7861>. Mặc định dùng **Fine-tune Extended 1 v4** qua `models/extended1_bundle.json`: checkpoint riêng cho tên, ngày sinh, tôn giáo; dân tộc giữ v2. Các checkpoint mới được fine-tune tiếp từ VietOCR chuyên chữ viết tay, có ghi nguồn ở [FINETUNE_V4.md](FINETUNE_V4.md). Có thể chọn model gốc hoặc **Fine-tune v3 · đối chiếu** rồi bấm OCR lại trên cùng crop; lựa chọn v3 chỉ có khi đủ trọng số lịch sử. Nếu chưa có bundle nhưng có `extended1_selected.pth`, script dùng lại v2. File trọng số chỉ có ở máy train; nhóm cần nhận riêng. Nếu cổng đã có demo đang chạy thì mở trang, không khởi động thêm.

Hai phiếu mẫu cũ vẫn có thể dùng để thử giao diện. Bộ 50 phiếu có cả bút xanh và bút đen. Kết quả so sánh model gốc và bản fine-tune trên tập test được ghi trong [EXTENDED1_STATUS.md](EXTENDED1_STATUS.md).

**Kết quả mới:** trên cùng 28 crop test, v3 đúng OCR thô 11/28, v4 đúng 13/28; CER giảm 24,7% → 13,8%. Sau định dạng/giải mã trường, tăng 15/28 → 19/28. Validation tăng 15/28 → 22/28, CER 13,6% → 3,95%. Đây là khảo sát trên tập test đã dùng nhiều lần, chưa biết ID người viết; vẫn còn lỗi và có từng phiếu bị kém đi. Chi tiết nguồn khởi tạo, train, checkpoint và giới hạn ở [FINETUNE_V4.md](FINETUNE_V4.md); lịch sử v3 ở [FINETUNE_V3.md](FINETUNE_V3.md).

## Chạy với hai ảnh mẫu

Trong thư mục repo:

```powershell
& '.\.venv\Scripts\python.exe' '.\prepare_dataset.py'
```

Kết quả nằm trong `prepared/`: mỗi phiếu có `aligned.png`, `debug_fields.png` và bốn ảnh trường; `to_label.csv` liệt kê toàn bộ ảnh cắt. Chạy lại lệnh sẽ giữ các cột `writer_id` và `label` đã điền.

## Thu và gán nhãn dữ liệu mới

1. In cùng một mẫu phiếu, nhờ **nhiều người khác nhau** điền. Ghi một mã người viết ổn định cho mọi phiếu của người đó. Chụp toàn bộ vùng phiếu rõ nét; giữ cả tiêu đề và các nhãn in để căn ảnh.
2. Đặt ảnh `.png`, `.jpg` hoặc `.jpeg` vào một thư mục riêng, ví dụ `forms/`, rồi chạy:

   ```powershell
   & '.\.venv\Scripts\python.exe' '.\prepare_dataset.py' --input-dir forms --output-dir prepared
   ```

3. Mở `prepared/to_label.csv`, điền `writer_id` và `label` chính xác cho **từng ảnh cắt**. Kiểm tra ảnh cắt bằng mắt; nếu trường bị cắt hoặc nhận nhầm vùng thì sửa ảnh/cấu hình trước khi gán nhãn. Không lấy dự đoán OCR làm nhãn nếu chưa kiểm tra.
4. Khi có ít nhất ba người viết và mọi nhãn đã được điền:

   ```powershell
   & '.\.venv\Scripts\python.exe' '.\export_vietocr.py' --data-dir prepared
   ```

   Lệnh tạo `train_annotation.txt`, `valid_annotation.txt`, `test_annotation.txt` theo định dạng `đường_dẫn_ảnh<TAB>nhãn` của VietOCR. Mọi phiếu của một người chỉ thuộc **một** tập. Dùng `prepared/` làm `data_root` khi cấu hình VietOCR. Giữ nguyên tập test để so sánh mô hình gốc và mô hình tinh chỉnh.

## Giới hạn hiện tại

- ZIP có 50 phiếu nhưng không có đáp án hay mã người viết. Đã đọc 193/200 trường (45 phiếu đủ, năm phiếu gán một phần); cần nhóm kiểm nhãn lại. Chia train/valid/test theo **phiếu**, chưa đảm bảo người viết độc lập.
- Họ tên và chữ số ngày sinh còn sai. Không coi định dạng hợp lệ hoặc điểm model cao là bằng chứng nội dung đã đúng.
- Căn phiếu có thể thất bại nếu ảnh mờ, mất tiêu đề/nhãn in, hoặc mẫu in khác. Trường hợp này được báo `SKIP`; không đưa ảnh lỗi vào tập train.
- Tọa độ `FIELDS` và `LEGACY_FIELDS` trong `prepare_dataset.py` chỉ dành cho **hai bản in hiện có**. Nếu đổi biểu mẫu phải hiệu chỉnh lại.

Tài liệu dự án VietOCR mô tả [định dạng nhãn train/test](https://github.com/pbcquoc/vietocr#c%C3%A1ch-t%E1%BA%A1o-file-traintest).

Baseline PaddleOCR là thử nghiệm riêng trong thư mục `DemoPaddleOCR` của máy phát triển; mã đó không nằm trong repo này.

## Kiểm thử

```powershell
& '.\.venv\Scripts\python.exe' -m unittest test_extended1 -v
# Khi demo cổng 7861 đang chạy:
& '.\.venv\Scripts\python.exe' '.\smoke_demo.py'
```
