# Trọng số cho demo

`pretrained_vgg_transformer.pth` là trọng số **VietOCR gốc** tải từ URL trong cấu hình chính thức của VietOCR (`https://vocr.vn/data/vietocr/vgg_transformer.pth`). File này dùng cho demo trước khi nhóm có dữ liệu và fine-tune; không phải model tùy biến của nhóm. File `.pth` được `.gitignore` bỏ qua. Nếu máy mới không có file này, VietOCR tải trọng số gốc từ URL trong `vgg_transformer.yml` ở lần OCR đầu tiên.

`extended1_selected.pth` là checkpoint v2 đang dùng trong demo, được sao từ `extended1_v2_continued.pth` bằng `select_extended1_model.py` dựa trên valid. CNN được khóa, transformer được cập nhật trên 137 crop train và tám crop replay của hai mẫu cũ; augmentation chỉ dùng ở train. Các thử nghiệm mở CNN không tốt hơn nên bị loại. Metadata, checksum và số liệu ở `extended1_selected.json`; báo cáo ở `EXTENDED1_STATUS.md`.

Chạy `.\start_extended1.ps1` để mở model v2 và lựa chọn so với model gốc. Các checkpoint `extended1_finetuned.pth`, `incremental_*` là lịch sử thử trước, không còn là model mặc định. Trọng số khoảng 145 MiB, cần chia sẻ riêng cho thành viên nhóm.

Các file `.pth` và JSON dự đoán chi tiết chỉ ở máy train. Phiếu có thể chứa thông tin cá nhân và model được học từ bộ nhỏ này, nên không tự đưa trọng số hoặc dữ liệu lên repo công khai.
