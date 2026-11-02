# OCR phiếu viết tay tiếng Việt · Extended 1

Mẫu hiện tại có bốn trường: họ tên, ngày sinh, dân tộc, tôn giáo. Pipeline căn ảnh chụp theo `phieu_mau.png` bằng SIFT + homography, cắt bốn vùng, tách mực xanh hoặc trừ mẫu in để lấy mực đen. Bản fine tune VietOCR và kết quả trên 50 phiếu được ghi tại [EXTENDED1_STATUS.md](EXTENDED1_STATUS.md). Kết quả OCR hiện còn cần người dùng kiểm tra và sửa.

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
2. Xem bốn ảnh crop. Nếu vùng nào sai, bấm **Sửa crop** ở trường đó, dùng công cụ **Crop** trên ảnh phiếu, bấm **Apply** trong khung ảnh, rồi bấm **Dùng crop vừa chọn**. Nếu căn tự động thất bại, sửa crop tay cả bốn trường.
3. Tích **Crop đúng** cho đủ bốn trường rồi bấm **Xác nhận crop và chạy VietOCR**.
4. Sửa trực tiếp nội dung nhận dạng nếu cần, sau đó bấm **Xem và tải JSON**. JSON giữ cả dự đoán gốc và nội dung đã xác nhận.

Demo mặc định dùng **VietOCR vgg_transformer pretrained**, chưa phải mô hình đã fine-tune. Repo không chứa file trọng số 145 MB; nếu chưa có `models/pretrained_vgg_transformer.pth`, VietOCR tải trọng số gốc từ URL trong `models/vgg_transformer.yml` ở lần OCR đầu tiên. Cần Internet cho lần tải này. Khi đã có trọng số fine-tune, chạy `ocr_review_app.py --weights đường_dẫn_đến_file.pth` để dùng trọng số mới. Toàn bộ demo chỉ phục vụ trên máy này (`127.0.0.1`), chạy bằng CPU nên lần OCR đầu có thể chậm.

Trên máy đã train Extended 1, chạy `.\start_extended1.ps1` rồi mở <http://127.0.0.1:7861>. Nếu PowerShell chặn file script, dùng lệnh Python trực tiếp trong [EXTENDED1_STATUS.md](EXTENDED1_STATUS.md). Trọng số `models/extended1_finetuned.pth` chỉ có ở máy đã train; nhóm cần nhận riêng file này nếu muốn chạy cùng model trên máy khác.

Hai phiếu mẫu cũ vẫn có thể dùng để thử giao diện. Với bộ 50 phiếu mới, nhánh tách bút đen đã được chạy trên ảnh thật. Kết quả so sánh mô hình gốc và bản fine tune trên tập test được ghi riêng trong [EXTENDED1_STATUS.md](EXTENDED1_STATUS.md).

**Extended 1:** Đã fine tune trên ảnh thật có cả hai màu mực. Xem [EXTENDED1_STATUS.md](EXTENDED1_STATUS.md) để biết dữ liệu, thông số train, kết quả valid/test và giới hạn của phép đánh giá.

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

- ZIP có 50 phiếu nhưng không có đáp án hay mã người viết. Đã gán nhãn lượt đầu cho 45 phiếu; cần nhóm kiểm tra lại. Chia train/valid/test hiện theo **phiếu**, chưa đảm bảo người viết độc lập.
- Trên 7 phiếu test, bản fine tune đúng toàn chuỗi 2/28 trường, VietOCR gốc 1/28. Lỗi ký tự giảm nhưng kết quả vẫn cần bước sửa tay. Chi tiết theo trường và màu mực ở [EXTENDED1_STATUS.md](EXTENDED1_STATUS.md).
- Căn phiếu có thể thất bại nếu ảnh mờ, mất tiêu đề/nhãn in, hoặc mẫu in khác. Trường hợp này được báo `SKIP`; không đưa ảnh lỗi vào tập train.
- Tọa độ `FIELDS` trong `prepare_dataset.py` chỉ dành cho **mẫu phiếu hiện có**. Nếu đổi biểu mẫu phải hiệu chỉnh lại.

Tài liệu dự án VietOCR mô tả [định dạng nhãn train/test](https://github.com/pbcquoc/vietocr#c%C3%A1ch-t%E1%BA%A1o-file-traintest).

Baseline PaddleOCR là thử nghiệm riêng trong thư mục `DemoPaddleOCR` của máy phát triển; mã đó không nằm trong repo này.
