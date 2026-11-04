# Extended 1 · kết quả thực chạy ngày 04/11/2026

**Cập nhật mới nhất: bundle v5.** Đã thử thêm các hướng fine-tune nhưng không có trọng số mới vượt v4; giữ cả bốn checkpoint v4 và cải thiện decoder dân tộc/tôn giáo. OCR thô test giữ 13/28, CER 13,8%; sau giải mã tăng 19 → 21/28. Valid sau giải mã tăng 23 → 24/28. Xem [FINETUNE_V5.md](FINETUNE_V5.md) cho bằng chứng, các nhánh bị loại và giới hạn; nguồn khởi tạo/fine-tune v4 ở [FINETUNE_V4.md](FINETUNE_V4.md).

Đã triển khai và tự kiểm thử luồng tải ảnh → căn/crop → xem ảnh xử lý chấm hoặc crop gốc → sửa crop tay → xác nhận crop → chọn VietOCR gốc/fine-tune → sửa nội dung → xuất JSON. **Nhận dạng tự động vẫn chưa đủ chính xác để bỏ bước người dùng kiểm tra.** Các phần dưới giữ lại hồ sơ của lượt v2 để đối chiếu; hướng dẫn model đang dùng xem tài liệu v5 ở trên.

## Thay đổi ở ảnh đầu vào

Giữ riêng tọa độ cho hai bản in hiện có. Nới vùng họ tên của bộ 50 phiếu để không cắt mất tên dài; loại các mảnh tiêu đề hoặc hàng kế tiếp bị lọt ở sát mép crop. Với bộ mới, cả mực xanh và đen đi qua cùng nhánh chuẩn hóa nền và lọc chấm bảo toàn nét chữ.

Nhánh mới chỉ xóa các chấm nhỏ tách rời nằm ngoài cụm chữ. Chấm dính vào nét viết, dấu tiếng Việt và chấm bên trong từ được giữ lại để recognizer học; không còn xóa cả dải ngang theo mẫu in. Vì vậy ảnh có thể vẫn còn vài chấm, nhưng tránh cắt đứt nét. Trong giao diện, checkbox **Lọc dòng chấm** cho từng trường giúp so ngay với crop gốc; thay ảnh sẽ hủy kết quả OCR cũ của trường đó và yêu cầu xác nhận lại crop.

## Dữ liệu và nhãn

- ZIP có 50 ảnh, tạo 200 crop. Nhãn được đọc trực tiếp từ ảnh, không có đáp án hay ID người viết do nhóm cung cấp.
- Đã rà lại 45 phiếu đủ bốn trường. Với năm phiếu từng bỏ ngoài, chỉ bổ sung 13 trường đọc rõ; bảy trường còn mơ hồ hoặc ngày không rõ/hợp lệ vẫn để trống. Tổng cộng **193 trường có nhãn**. Đây là một lượt kiểm tra bằng mắt của trợ lý, chưa thay thế việc hai thành viên nhóm đối chiếu lại.
- Giữ nguyên bảy nhóm valid và bảy nhóm test cũ; năm phiếu gán nhãn một phần chỉ thêm vào train. Train: **36 phiếu / 137 crop**; valid: **7 / 28**; test: **7 / 28**.
- Hai phiếu demo cũ / tám crop được thêm làm dữ liệu replay trong train để giữ khả năng đọc mẫu cũ. Số đúng 8/8 trên hai phiếu này là kiểm tra hồi quy trên dữ liệu train, không phải kết quả tổng quát hóa.
- Train có ba phiên bản ảnh mỗi crop: ảnh sạch cùng các biến thể nhẹ về độ nghiêng, độ rộng, tương phản, nhòe và chấm giả. Các biến thể không sang valid/test và không được tính thành người viết hay phiếu mới.

Nhãn cục bộ: `forms/dataset_20261102/ground_truth_audited_v2.csv`; ghi chú kiểm nhãn: `label_audit_v2.json`. Crop/annotation mới: `prepared/extended1_v2/`. Cấu hình chia nhóm có thể tái sử dụng ở `configs/extended1_split.json`.

**Giới hạn đánh giá:** `form_001`… là ID phiếu tạm, không phải người viết. Chưa loại trừ một người xuất hiện ở nhiều tập. Tập test đã được khảo sát trong các lần thử cũ và toàn bộ bộ ảnh đã được xem khi chỉnh preprocessing. Các số dưới đây là đánh giá khảo sát trên dữ liệu hiện có, chưa phải kiểm định độc lập trên người viết mới.

## Fine-tune và lựa chọn checkpoint

Giữ kiến trúc VietOCR `vgg_transformer`, backbone `vgg19_bn`, thư viện VietOCR `0.3.13`; khởi đầu từ `pretrained_vgg_transformer.pth`. CNN được khóa trong lượt được chọn; transformer được fine-tune với label smoothing, augmentation chỉ trên train, gradient clipping và early stopping. Cache đặc trưng CNN giúp train được trên CPU.

| Giai đoạn | Kết quả valid tốt nhất | CER valid | Hai mẫu replay |
| --- | ---: | ---: | ---: |
| Model gốc, cùng preprocessing mới | 3/28 | 41,2% | 7/8 |
| Lượt đầu, LR `3e-5`, dừng ở epoch 5 theo yêu cầu pause | 10/28 | 21,1% | 8/8 |
| Tiếp tục từ checkpoint trên, LR `2e-5`, chọn epoch 8 của lượt tiếp | **13/28** | **15,4%** | **8/8** |

Lượt tiếp chạy đến epoch 15 rồi dừng sớm vì valid không tốt hơn. Khi tiếp tục từ file trọng số, optimizer được khởi tạo lại; đây không phải khôi phục nguyên trạng optimizer trước pause. Hai phép thử mở khối cuối CNN và một phép thử beam search không tốt hơn trên valid nên bị loại. Checkpoint được chọn trước khi xem kết quả test của lượt này.

File nguồn được chọn: `models/extended1_v2_continued.pth`. File demo: `models/extended1_selected.pth`, khoảng 145 MiB. SHA-256: `3758af1dda76597061b8b4bb8fdf49b97a632a293875744e4c4377b642ddd250`. Metadata và số liệu đối chiếu ở `models/extended1_selected.json`.

## So sánh trên cùng 28 crop test

CER tính số thao tác sửa ký tự chia tổng ký tự nhãn. OCR thô tính đúng toàn bộ chuỗi, phân biệt dấu và chữ hoa/thường. Cột định dạng chỉ chuẩn hóa ngày số hợp lệ và bỏ khác biệt hoa/thường. Cột có ràng buộc bổ sung giải mã theo khuôn ngày và chấm điểm chuỗi ứng viên dân tộc/tôn giáo bằng VietOCR; không có sửa tay trong các số này.

| Model | OCR thô đúng toàn trường | CER thô | Sau định dạng | Có ràng buộc trường |
| --- | ---: | ---: | ---: | ---: |
| VietOCR gốc | 2/28 | 45,6% | 2/28 | 4/28 |
| Fine-tune v2 | **10/28** | **25,5%** | **12/28** | **14/28** |

OCR thô trên mực đen: gốc 1/12, fine-tune 6/12; CER 39,6% → 20,8%. Mực xanh: gốc 1/16, fine-tune 4/16; CER 50,0% → 29,0%. Họ tên của fine-tune mới đúng 2/7; ngày sinh còn nhầm chữ số. Trên valid, kết quả có ràng buộc đạt 16/28 so với OCR thô 13/28.

Các đề xuất có ràng buộc đều giữ OCR thô và cảnh báo trong JSON. Chúng vẫn có thể sai: một ngày được ép về đúng khuôn có thể hợp lệ về lịch nhưng sai nội dung. Ví dụ phiếu 50 hiện đọc tháng 11 thành 01, dù họ tên, dân tộc và tôn giáo đúng sau giải mã trường. Không coi `all_fields_valid` là độ chính xác; đó chỉ là kiểm tra định dạng.

Trang đối chiếu cục bộ: `prepared/extended1_v2/results.html` (ảnh phiếu, crop gốc/sạch, nhãn, kết quả hai model). Báo cáo từng dự đoán: `models/extended1_v2_test_structured.json` và `models/extended1_v2_valid_structured.json`.

## Chạy và tự kiểm tra

Trong PowerShell tại thư mục repo:

```powershell
.\start_extended1.ps1
```

Mở <http://127.0.0.1:7861/>. Model mặc định là **Fine-tune Extended 1**, có thể chọn VietOCR gốc để so cùng crop. Nếu đã có tiến trình chạy cổng đó thì mở trang, không khởi động thêm một bản.

Kiểm thử đã qua: bảy test về giữ nét qua dòng chấm, dấu tiếng Việt, định dạng ngày không đoán chữ số, tách nhóm phiếu, train/inference cùng preprocessing, đổi crop hủy dự đoán cũ và JSON giữ OCR thô/nội dung sửa. Kiểm thử Gradio thật đã chạy cả model gốc/fine-tune, bật/tắt lọc, chặn OCR chưa xác nhận, crop tay và tải JSON. Phiếu bút đen số 1 trong train đọc 4/4; phiếu bút xanh số 50 trong test đọc 3/4 sau giải mã trường. Đã mở và thao tác trên giao diện thật.

```powershell
& '.\.venv\Scripts\python.exe' -m unittest test_extended1 -v
& '.\.venv\Scripts\python.exe' '.\smoke_demo.py'
```

## Tái hiện quy trình với dữ liệu cục bộ

```powershell
& '.\.venv\Scripts\python.exe' '.\prepare_dataset.py' --input-dir '.\forms\dataset_20261102' --output-dir '.\prepared\extended1_v2' --ground-truth '.\forms\dataset_20261102\ground_truth_audited_v2.csv'
& '.\.venv\Scripts\python.exe' '.\export_vietocr.py' --data-dir '.\prepared\extended1_v2' --skip-unlabeled --allow-provisional-form-groups --reuse-split '.\configs\extended1_split.json'
& '.\.venv\Scripts\python.exe' '.\prepare_dataset.py' --input-dir '.' --output-dir '.\prepared\extended1_v2\pilot' --ground-truth '.\pilot_labels.csv'
& '.\.venv\Scripts\python.exe' '.\finetune_v2.py' --output '.\models\extended1_v2.pth' --epochs 30 --stop-after 5 --lr 0.00003
& '.\.venv\Scripts\python.exe' '.\finetune_v2.py' --initial-weights '.\models\extended1_v2.pth' --output '.\models\extended1_v2_continued.pth' --epochs 25 --lr 0.00002 --patience 7
& '.\.venv\Scripts\python.exe' '.\evaluate_test.py' --data-dir '.\prepared\extended1_v2' --weights '.\models\extended1_v2_continued.pth' --output '.\models\extended1_v2_valid_structured.json' --split valid --structured
& '.\.venv\Scripts\python.exe' '.\evaluate_test.py' --data-dir '.\prepared\extended1_v2' --weights '.\models\extended1_v2_continued.pth' --output '.\models\extended1_v2_test_structured.json' --split test --structured
& '.\.venv\Scripts\python.exe' '.\select_extended1_model.py'
& '.\.venv\Scripts\python.exe' '.\make_review_page.py' --data-dir '.\prepared\extended1_v2' --source-dir '.\forms\dataset_20261102' --evaluation '.\models\extended1_v2_test_structured.json'
```

Ảnh, nhãn và trọng số không nằm trên GitHub; nhóm cần nhận riêng các file này. Lần đánh giá chính thức tiếp theo cần nhãn được nhóm kiểm lại, ID người viết thật và phiếu mới chưa dùng để chỉnh pipeline. Tài liệu tham khảo: [VietOCR](https://github.com/pbcquoc/vietocr); danh mục dân tộc dùng cho giải mã, có thêm biến thể tên gọi, tham khảo [InvestVietnam](https://investvietnam.gov.vn/vi/co-ban-ve-viet-nam.nd/cac-dan-toc-viet-nam.html). Danh sách tôn giáo trong demo chỉ là các giá trị ví dụ hỗ trợ, không phải một phân loại đầy đủ.
