# Trọng số cho demo

**Bản đang chạy là v4:** `extended1_bundle.json` chọn bốn checkpoint bên dưới. Mỗi file khoảng 145 MiB; cần nhận đủ bốn file và bundle khi chạy trên máy khác. Bundle kiểm tra checksum, ghi nguồn khởi tạo và chọn model theo trường. Xem [FINETUNE_V4.md](../FINETUNE_V4.md) cho số liệu, attribution và lệnh tái hiện.

| Trường | Checkpoint |
| --- | --- |
| Họ tên | `extended1_v4_handwriting_adapt_ho_ten.pth` |
| Ngày sinh | `extended1_v4_handwriting_adapt_ngay_sinh.pth` |
| Dân tộc | `extended1_v2_continued.pth` |
| Tôn giáo | `extended1_v4_dung_adapt.pth` |

V3 còn ở `extended1_v3_previous_bundle.json`; cần `extended1_v2_continued.pth` và `extended1_v3_date_visual.pth` để chạy lại v3. `extended1_selected.pth` bên dưới vẫn giữ làm mốc/fallback v2.

`pretrained_vgg_transformer.pth` là trọng số **VietOCR gốc** tải từ URL trong cấu hình chính thức của VietOCR (`https://vocr.vn/data/vietocr/vgg_transformer.pth`). File này dùng cho demo trước khi nhóm có dữ liệu và fine-tune; không phải model tùy biến của nhóm. File `.pth` được `.gitignore` bỏ qua. Nếu máy mới không có file này, VietOCR tải trọng số gốc từ URL trong `vgg_transformer.yml` ở lần OCR đầu tiên.

`extended1_selected.pth` là checkpoint v2 giữ lại để đối chiếu/fallback, được sao từ `extended1_v2_continued.pth` bằng `select_extended1_model.py` dựa trên valid. CNN được khóa, transformer được cập nhật trên 137 crop train và tám crop replay của hai mẫu cũ; augmentation chỉ dùng ở train. Metadata, checksum và số liệu ở `extended1_selected.json`; báo cáo v2 ở `EXTENDED1_STATUS.md`.

Chạy `.\start_extended1.ps1` để mở bundle v4 và lựa chọn so với model gốc; script quay về v2 nếu chưa có bundle. Các checkpoint `extended1_finetuned.pth`, `incremental_*` là lịch sử thử trước, không còn là model mặc định.

Các file `.pth` và JSON dự đoán chi tiết chỉ ở máy train. Phiếu có thể chứa thông tin cá nhân và model được học từ bộ nhỏ này, nên không tự đưa trọng số hoặc dữ liệu lên repo công khai.
