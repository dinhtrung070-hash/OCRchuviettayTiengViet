# Trọng số cho demo

`pretrained_vgg_transformer.pth` là trọng số **VietOCR gốc** tải từ URL trong cấu hình chính thức của VietOCR (`https://vocr.vn/data/vietocr/vgg_transformer.pth`). File này dùng cho demo trước khi nhóm có dữ liệu và fine-tune; không phải model tùy biến của nhóm. File `.pth` được `.gitignore` bỏ qua. Nếu máy mới không có file này, VietOCR tải trọng số gốc từ URL trong `vgg_transformer.yml` ở lần OCR đầu tiên.

Khi có trọng số fine-tune, chạy app với `--weights <đường-dẫn-file.pth>`. Giao diện sẽ ghi rõ model đang dùng là bản fine-tune.
