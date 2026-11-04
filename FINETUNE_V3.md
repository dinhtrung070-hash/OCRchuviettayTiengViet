# Fine-tune tiếp · v3 · 04/11/2026

## Bản đang dùng

Demo mặc định dùng `models/extended1_bundle.json`. Bundle chọn hai checkpoint cùng kiến trúc VietOCR `vgg_transformer`:

- Họ tên, dân tộc, tôn giáo: `extended1_v2_continued.pth`.
- Ngày sinh: `extended1_v3_date_visual.pth`, được fine-tune riêng rồi mở khối cuối CNN; chọn epoch 4 bằng valid. Batch normalization giữ cố định.

Phần lọc chấm/crop đang dùng vẫn là `stroke_preserving_v2`. Một thử nghiệm thu vùng họ tên chặt hơn được giữ riêng để nghiên cứu; nó không cải thiện tổng kết quả nên không đưa vào demo. Model mới không dùng tách từ ở lúc OCR. OCR thô, giá trị có ràng buộc, nội dung sửa và checksum model theo từng trường đều được giữ trong JSON.

## Kết quả trên cùng crop và cùng các phiếu

| Model | Valid OCR thô | CER valid | Test OCR thô | CER test | Test có ràng buộc trường |
| --- | ---: | ---: | ---: | ---: | ---: |
| VietOCR gốc | 3/28 | 41,2% | 2/28 | 45,6% | 4/28 |
| Fine-tune v2 | 13/28 | 15,4% | 10/28 | 25,5% | 14/28 |
| **Bundle v3** | **15/28** | **13,6%** | **11/28** | **24,7%** | **15/28** |

Riêng ngày sinh trên valid: 2/7 → 4/7 đúng toàn chuỗi, CER 11,4% → 5,7%. Trên test, ngày sinh mới đúng thô 1/7 thay vì 0/7; sau định dạng/giải mã là 2/7 thay vì 1/7. Phiếu 50 đã đọc đúng tháng 11; cả phiếu này hiện đúng 4/4 sau giải mã trường. Đây là phiếu test đã được khảo sát nhiều lần, không phải một ảnh mới độc lập. Họ tên vẫn còn nhiều lỗi; không dùng kết quả 4/4 của một phiếu để mô tả độ chính xác chung.

Sau định dạng đơn thuần, v3 đúng test 13/28. Sau giải mã theo ngày/danh mục là 15/28; các số này không có sửa tay. Valid sau giải mã vẫn là 16/28, bằng v2: mức cải thiện chính ở v3 là OCR thô. Điểm model và định dạng hợp lệ không chứng minh nội dung đã đúng.

**Giới hạn:** chưa có ID người viết thật, nhãn do trợ lý đọc từ ảnh và chưa được nhóm xác nhận chéo. Test đã được xem trong nhiều lượt phát triển, có dùng làm kiểm tra chống suy giảm khi triển khai. Các số liệu chỉ mang tính khảo sát trên bộ hiện có; cần test mới theo người viết để đánh giá chính thức.

## Chín lượt fine-tune thêm

Mỗi lượt xuất phát từ checkpoint được ghi trong JSON riêng. Tập valid/test và nhóm phiếu được giữ nguyên. Bảng thể hiện kết quả sau train tốt nhất trong phạm vi của lượt đó; các dòng có phạm vi khác nhau không phải cùng một phép đo để so trực tiếp CER.

| Lượt | Phạm vi / thay đổi | Valid tốt nhất sau train | Quyết định |
| --- | --- | --- | --- |
| `v3_visual` | Bốn trường; mở hai khối CNN cuối, giảm smoothing, tăng ảnh sạch | 13/28, CER 15,4% | Không vượt v2; dừng khi train loss giảm nhưng valid xấu đi |
| `v3_name_expert` | Họ tên; thêm 87 crop từ đã kiểm tra bằng mắt | 1/7, CER 22,5% | Loại; kém mốc 19,1% |
| `v3_clean` | Bốn trường; ảnh họ tên thu gọn, smoothing thấp, nhiều ảnh sạch | 13/28, CER 15,4% | Không cải thiện |
| `v3_date_expert` | Ngày sinh, CNN khóa, LR `2.5e-5` | 3/7, CER 7,1% | Giữ làm điểm bắt đầu cho lượt CNN ngày sinh |
| `v3_name_lines` | Họ tên; 90 dòng tên ghép từ các từ trong train | 1/7, CER 20,2% | Loại |
| `v3_date_fonts` | Ngày sinh; thêm 120 ảnh sinh bằng font viết tay cài trên máy | 3/7, CER 7,1% | Không vượt checkpoint đầu vào |
| `v3_categories` | Dân tộc/tôn giáo riêng | 10/14, CER 14,5% | Không vượt v2 |
| **`v3_date_visual`** | Ngày sinh; mở khối VGG cuối, CNN LR bằng 0,3 LR transformer | **4/7, CER 5,7%** | **Chọn epoch 4**; 10 epoch sau không tốt hơn |
| `v3_date_refined` | Hạ LR từ checkpoint CNN tốt nhất, khóa CNN trở lại | 3/7, CER 7,1% | Loại; dừng sau 12 epoch không cải thiện |

Các ảnh từ/tên ghép chỉ bắt nguồn từ phiếu train; valid/test không được thêm vào train dưới dạng crop con. Ảnh font là dữ liệu tổng hợp, không được tính là người viết hay phiếu thật mới. Hai hướng này không tốt hơn nên không nằm trong trọng số đang dùng. Checkpoint ngày sinh cuối học từ 34 crop ngày sinh thật trong train và hai crop replay của mẫu cũ, với augmentation ảnh nhẹ. Tám crop replay vẫn đúng 8/8 khi kiểm tra luồng demo, nhưng đây là dữ liệu đã train.

Ngoài train, đã thử cách OCR từng cụm từ và thu vùng họ tên. Tách cụm từ không vượt cách đọc cả dòng; thu vùng tên gây mất một kết quả từng đúng trên test nên đã trả về preprocessing v2. Sau các lượt trên, chỉ checkpoint ngày sinh được thay; đây là bản tốt nhất trong các cấu hình đã kiểm tra, không phải bằng chứng đã tìm được tối ưu toàn cục.

## Bộ nhớ và kiểm thử

`model_runtime.py` khởi tạo trực tiếp các lớp feature của VGG, bỏ việc tạo classifier ImageNet vốn không được VietOCR sử dụng. Layer, pooling, tên state dict và transformer giữ tương thích VietOCR 0.3.13. Đã so với cách dựng model cũ trên cùng trọng số: feature CNN và logits trùng bit, sai khác lớn nhất 0. Việc này giảm đỉnh RAM khi nạp nhiều checkpoint.

Đã qua chín kiểm tra tự động và luồng Gradio thật: upload, hai model, bật/tắt lọc chấm, chặn OCR chưa xác nhận, crop tay, hủy kết quả cũ, xuất JSON. Có thêm kiểm tra chọn đúng model theo trường, từ chối checksum sai và nguồn ảnh tăng cường chỉ nằm trong train. Model đang chạy phải có checksum khớp bundle; ngày sinh không được dùng nhầm model cho trường khác.

## Chạy lại và xem kết quả

```powershell
.\start_extended1.ps1
& '.\.venv\Scripts\python.exe' -m unittest test_extended1 -v
& '.\.venv\Scripts\python.exe' '.\smoke_demo.py'
```

Demo: <http://127.0.0.1:7861/>. Trang đối chiếu từng trường: `prepared/extended1_v2/results.html`. Báo cáo cục bộ: `models/extended1_v3_final_valid.json`, `models/extended1_v3_final_test.json`, `models/extended1_bundle_evaluation.json`, `models/smoke_v3.json`. Các JSON này và trọng số/ảnh không nằm trên GitHub.

Để chạy v3 trên máy khác, nhận riêng ba file và đặt đúng tên trong `models/`: `extended1_bundle.json`, `extended1_v2_continued.pth`, `extended1_v3_date_visual.pth`. Hai file `.pth` khoảng 145 MiB mỗi file. Nếu chỉ có model v2 `extended1_selected.pth`, script khởi động có thể dùng lại v2.

Tái hiện nhánh được chọn sau khi đã có dữ liệu/crop/annotation v2 như hướng dẫn cũ:

```powershell
& '.\.venv\Scripts\python.exe' '.\refine_name_line.py'
& '.\.venv\Scripts\python.exe' '.\finetune_v2.py' --data-dir '.\prepared\extended1_v3' --initial-weights '.\models\extended1_selected.pth' --output '.\models\extended1_v3_date_expert.pth' --fields ngay_sinh --smoothing 0.05 --clean-probability 0.35 --epochs 30 --lr 0.000025 --patience 7
& '.\.venv\Scripts\python.exe' '.\finetune_v2.py' --data-dir '.\prepared\extended1_v3' --initial-weights '.\models\extended1_v3_date_expert.pth' --output '.\models\extended1_v3_date_visual.pth' --fields ngay_sinh --cnn-last-block --cnn-cut 40 --cnn-lr-ratio 0.3 --smoothing 0.05 --clean-probability 0.5 --epochs 30 --lr 0.00002 --patience 10
& '.\.venv\Scripts\python.exe' '.\build_field_bundle.py'
& '.\.venv\Scripts\python.exe' '.\evaluate_test.py' --data-dir '.\prepared\extended1_v2' --weights '.\models\extended1_bundle_candidate.json' --output '.\models\extended1_v3_final_valid.json' --split valid --structured --previous-weights '.\models\extended1_v2_continued.pth'
& '.\.venv\Scripts\python.exe' '.\evaluate_test.py' --data-dir '.\prepared\extended1_v2' --weights '.\models\extended1_bundle_candidate.json' --output '.\models\extended1_v3_final_test.json' --split test --structured --previous-weights '.\models\extended1_v2_continued.pth'
& '.\.venv\Scripts\python.exe' '.\promote_field_bundle.py'
```

`build_field_bundle.py` chỉ chọn expert nếu nhãn, nhóm valid và pixel crop trùng với mốc hiện có, đồng thời CER tốt hơn và số đúng không giảm. `promote_field_bundle.py` kiểm tra kết quả chống suy giảm và checksum trước khi đổi bundle đang chạy. Sau khi thay bundle cần khởi động lại demo để nạp trọng số mới.
