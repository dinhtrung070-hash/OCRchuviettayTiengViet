# OCR phiếu viết tay: demo và chuẩn bị dữ liệu cho Extended 1

Mẫu hiện tại có bốn trường: họ tên, ngày sinh, dân tộc, tôn giáo. Pipeline căn ảnh chụp theo `phieu_mau.png` bằng SIFT + homography, cắt bốn vùng, và tách nét bút xanh khỏi chữ in/dòng chấm. Ảnh `debug_fields.png` cho phép kiểm tra vị trí cắt trước khi gán nhãn.

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

Thử trực tiếp hai phiếu hiện có: phiếu `phieudaviet1.png` được nhận đúng 4/4 trường trong một lần chạy; phiếu `phieu_da_viet.png` bị sai trường **tôn giáo**. Đây là kết quả thử hai ảnh, chưa phải số đo độ chính xác. Khi không phát hiện mực xanh, giao diện thử trừ nền mẫu để lấy nét bút đen; nếu không đủ nét thì giữ crop gốc. Cách xử lý này chưa được kiểm chứng trên phiếu viết bút đen thật.

**Pilot Extended 1:** Xem [EXTENDED1_STATUS.md](EXTENDED1_STATUS.md) để chạy fine-tune thử trên hai phiếu, kiểm tra bản mô phỏng mực đen và dùng trọng số mới trong giao diện. Kết quả hiện tại chỉ là kiểm thử trên dữ liệu đã dùng để train; chưa có số đo cho người viết và bút đen mới.

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

- Chỉ kiểm tra trực quan trên **hai phiếu mẫu**; chưa có đủ dữ liệu để huấn luyện hoặc báo cáo độ chính xác.
- Bộ tách mực ưu tiên bút xanh. Nếu không tìm thấy nét xanh, chương trình thử trừ nền mẫu để lấy nét tối; khi vẫn không đủ nét thì giữ crop gốc và báo `blue_ink_detected=False`. Bút đen thật cần được kiểm tra sau khi có dữ liệu mới.
- Căn phiếu có thể thất bại nếu ảnh mờ, mất tiêu đề/nhãn in, hoặc mẫu in khác. Trường hợp này được báo `SKIP`; không đưa ảnh lỗi vào tập train.
- Tọa độ `FIELDS` trong `prepare_dataset.py` chỉ dành cho **mẫu phiếu hiện có**. Nếu đổi biểu mẫu phải hiệu chỉnh lại.

Sau khi có dữ liệu gán nhãn, bước tiếp theo là tinh chỉnh bộ nhận dạng rồi chạy cùng một tập test cho mô hình gốc và mô hình mới. Giao diện demo đã có sẵn để so sánh trực tiếp và xuất JSON. Tài liệu dự án VietOCR mô tả [định dạng nhãn train/test](https://github.com/pbcquoc/vietocr#c%C3%A1ch-t%E1%BA%A1o-file-traintest).

Baseline PaddleOCR là thử nghiệm riêng trong thư mục `DemoPaddleOCR` của máy phát triển; mã đó không nằm trong repo này.
