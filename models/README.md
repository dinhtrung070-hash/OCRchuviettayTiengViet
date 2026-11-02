# Trọng số cho demo

`pretrained_vgg_transformer.pth` là trọng số **VietOCR gốc** tải từ URL trong cấu hình chính thức của VietOCR (`https://vocr.vn/data/vietocr/vgg_transformer.pth`). File này dùng cho demo trước khi nhóm có dữ liệu và fine-tune; không phải model tùy biến của nhóm. File `.pth` được `.gitignore` bỏ qua. Nếu máy mới không có file này, VietOCR tải trọng số gốc từ URL trong `vgg_transformer.yml` ở lần OCR đầu tiên.

`extended1_finetuned.pth` là checkpoint được chọn sau khi fine tune trên 45 phiếu được gán nhãn lượt đầu từ `Dataset.zip` ngày 02/11/2026. CNN được giữ nguyên và transformer được cập nhật; kết quả valid/test ở `EXTENDED1_STATUS.md`. Chạy app với `--weights models/extended1_finetuned.pth` hoặc `.\start_extended1.ps1`. Giao diện sẽ ghi rõ model đang dùng là bản fine tune.

Các file `.pth` và JSON dự đoán chi tiết chỉ ở máy train. Phiếu có thể chứa thông tin cá nhân và model được học từ bộ nhỏ này, nên không tự đưa trọng số hoặc dữ liệu lên repo công khai.
