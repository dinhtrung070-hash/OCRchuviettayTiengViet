# Fine-tune tiếp · v4 · 04/11/2026

## Kết quả đối chiếu cuối

Đã fine-tune tiếp hai trọng số VietOCR chữ viết tay trên dữ liệu của nhóm và chọn checkpoint theo từng trường. Giữ nguyên pixel crop, cách chia dữ liệu, nhãn và preprocessing v2 để chạy đối chiếu với v3. Đây là bản tốt nhất trong đợt thử v4, không phải bằng chứng tối ưu toàn cục.

| Bản | Valid OCR thô | CER valid | Test OCR thô | CER test | Test sau định dạng | Test sau giải mã trường |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VietOCR gốc | 3/28 | 41,2% | 2/28 | 45,6% | 2/28 | 4/28 |
| Bundle v3 | 15/28 | 13,6% | 11/28 | 24,7% | 13/28 | 15/28 |
| **Bundle v4** | **22/28** | **3,95%** | **13/28** | **13,8%** | **16/28** | **19/28** |

Valid v4 sau định dạng/giải mã đạt 23/28. Các số sau giải mã không có sửa tay, nhưng có dùng khuôn ngày và danh mục dân tộc/tôn giáo; chúng không phải độ chính xác OCR thô. Họ tên không dùng từ điển đáp án. Test đã được dùng trong nhiều lượt phát triển, nên chỉ mang tính khảo sát.

| Trường trên test | V3 OCR thô | V4 OCR thô | CER v3 → v4 | Sau giải mã v3 → v4 |
| --- | ---: | ---: | ---: | ---: |
| Họ tên | 2/7 | 3/7 | 29,8% → 11,7% | 2/7 → 3/7 |
| Ngày sinh | 1/7 | 2/7 | 24,3% → 18,6% | 2/7 → 4/7 |
| Dân tộc | 4/7 | 4/7 | 12,5% → 12,5% | 5/7 → 5/7 |
| Tôn giáo | 4/7 | 4/7 | 21,6% → 11,8% | 6/7 → 7/7 |

Theo phân loại mực của pipeline trên test: xanh có 16 trường, OCR thô tăng 5→6, CER 27,5%→13,0%, sau giải mã 8→11; đen/tối có 12 trường, OCR thô 6→7, CER 20,8%→14,9%, sau giải mã 7→8. Đây là phân loại tự động, chưa có nhãn mực độc lập do nhóm xác nhận.

**Lỗi còn lại:** phiếu 4 vẫn sai tên và chữ số ngày sinh; phiếu 50 bị thừa ký tự `M` cuối họ tên trong khi v3 từng đọc đúng trường đó. Vì vậy cải thiện tổng thể không đồng nghĩa từng phiếu đều tốt hơn. Ngày hợp lệ cũng có thể sai nội dung. Chênh lệch valid/test còn lớn; vẫn phải đối chiếu ảnh và sửa kết quả trước khi dùng.

## Cách đánh giá và phạm vi dữ liệu

Dữ liệu giữ nguyên cách chia v2: 137 crop train từ 36 nhóm phiếu tạm thời, 28 crop validation từ 7 nhóm và 28 crop test từ 7 nhóm. Tám crop của hai mẫu cũ được dùng làm replay trong train. Kết quả trên replay chỉ kiểm tra chống quên mẫu cũ, không phải độ chính xác trên dữ liệu độc lập.

Tên nhóm phiếu chưa chứng minh đó là những người viết khác nhau; chưa có ID người viết thật để bảo đảm chia theo người viết. Nhãn là bản đọc ảnh lần đầu và còn cần nhóm xác nhận chéo. Không đưa crop validation/test hoặc các phần cắt nhỏ của chúng vào train. Các thử nghiệm mới ở giai đoạn lựa chọn chỉ đọc train/validation; bộ test cũ được giữ cho bước đối chiếu cuối, không dùng để chọn epoch.

## Những hướng thử ban đầu không cải thiện

Đã kiểm thử lại demo v3 và khảo sát các lỗi còn lại. Những thay đổi dưới đây chưa đem lại cải thiện đủ để đưa vào hệ thống.

| Hướng thử | Quan sát trên validation | Quyết định |
| --- | --- | --- |
| Beam search rộng 8, có/không giới hạn bảng chữ tên, ba mức chuẩn hóa độ dài | Tên vẫn 1/7; CER 20,2%, so với 19,1% của greedy. Ngày sinh vẫn 4/7, CER 5,7% | Giữ greedy |
| Nới khoảng cách/đổi ngưỡng điểm và biên giữa hai đáp án danh mục | Dân tộc + tôn giáo vẫn 11/14 sau giải mã trong các cấu hình đã kiểm tra | Không ép đáp án khi hình ảnh chưa đủ rõ |
| Thu khoảng trắng theo chiều dọc, thêm đệm, đổi tỷ lệ ngang 0,85/1,15, gom cụm chữ lớn cho tên | Không cấu hình nào vượt 1/7 và CER 19,1% của crop gốc | Giữ hình học crop v2 |
| Chỉ fine-tune khối CNN cuối cho tên, khóa transformer | Sáu epoch; tốt nhất sau train 1/7, CER 20,2%; không vượt mốc 19,1% | Loại checkpoint thử nghiệm |
| Nội suy trọng số VietOCR gốc với v2, tỷ lệ v2 0,25/0,5/0,75 | Tên đều 1/7, CER lần lượt 40,4% / 32,6% / 27,0% | Không dùng |

Thử nghiệm tiền xử lý được đo trên cùng 28 trường validation và checkpoint v3:

| Tiền xử lý | Đúng OCR thô | CER | Đúng sau giải mã trường |
| --- | ---: | ---: | ---: |
| **V2 đang dùng** | **15/28** | **13,6%** | **16/28** |
| Trích cường độ mực xanh từ chroma; mực đen giữ cách cũ | 13/28 | 15,4% | 16/28 |
| Lọc median 3 | 13/28 | 15,8% | 16/28 |
| Tăng tương phản CLAHE nhẹ | 14/28 | 15,4% | 15/28 |
| Xóa thêm chấm rời trong khoảng cách giữa chữ | 15/28 | 14,0% | 16/28 |
| Mặt nạ chroma, giữ cường độ nét xám gốc | 14/28 | 15,4% | 16/28 |

Không chọn ảnh chỉ vì nhìn trắng/sạch hơn. Trích chroma làm một số nét mực xanh bị loang/mất thông tin; tăng tương phản khuếch đại cả nhiễu; xóa thêm chấm có thể làm khác phân bố ảnh model đã học. Phần xử lý đang dùng vẫn là `stroke_preserving_v2`, hỗ trợ cả mực đen và xanh. Chỉ xóa chấm nhỏ rời có bằng chứng đủ chắc; giữ dấu và thành phần dính nét chữ. Chữ viết và chấm in cùng màu, chạm vào nhau thì không thể bảo đảm tách hoàn hảo bằng một quy tắc ảnh đơn giản.

Các báo cáo cục bộ tương ứng: `models/v4_decode_valid.json`, `v4_category_valid.json`, `v4_name_geometry_valid.json`, `v4_preprocess_valid.json`, `extended1_v4_name_visual.json`, `extended1_v4_name_interpolation.json`. Những JSON này không nằm trên GitHub.

## Dữ liệu ngày sinh tổng hợp đã chuẩn bị nhưng chưa dùng

`v4_date_glyphs.py` chỉ đọc 34 crop ngày sinh trong train. Không có dòng nào tách được 10 ký tự rõ bằng khoảng trắng toàn chiều cao. Script dùng phần thân phía trên để đề xuất biên, rồi giữ pixel toàn chiều cao cho từng glyph; không huấn luyện bằng ảnh đã che baseline.

Sau kiểm tra bằng mắt, giữ 45 glyph từ 5 phiếu train, loại các vị trí cắt qua nét giao không rõ; ghép 100 ngày hợp lệ ngẫu nhiên. Pool chưa có chữ số 5 và 6 nên không sinh ngày cần hai chữ số này. Nguồn từng ký tự và ảnh ghép được lưu trong `prepared/extended1_v2/glyph_augmentation/audit.json`; contact sheet có tên `glyph_review.png` và `synthetic_review_*.png`.

**Chưa dùng nhánh glyph này để train hoặc tính thành dữ liệu mới.** Đây chỉ là ảnh tăng cường từ người viết đã có trong train; không phải 100 phiếu thật hay 100 người viết mới. Các nhánh thích nghi chữ viết tay bên dưới hiện dùng crop thật và augmentation ảnh nhẹ, không dùng nhãn bổ sung `glyph_synthetic_annotation.txt`.

## Đổi trọng số khởi tạo, vẫn fine-tune trên dữ liệu của nhóm

VietOCR hiện tại có kiến trúc `vgg_transformer`, VGG19-BN và transformer. Đã kiểm tra thêm hai trọng số cùng kiến trúc dành cho chữ viết tay. Đây là **trọng số khởi tạo từ bên ngoài**, không nhận là thành quả nhóm tự huấn luyện. Phần nhóm thực hiện là tiếp tục fine-tune, chọn checkpoint bằng validation trên dữ liệu phiếu và đánh giá pipeline của nhóm.

Nguồn và giấy phép theo model card:

| Nguồn | Revision cố định | Tệp tải về | Giấy phép ghi trên model card |
| --- | --- | --- | --- |
| [vudang449/vietocr-vietnamese-handwriting](https://huggingface.co/vudang449/vietocr-vietnamese-handwriting) | `f1973b930f907c378f6e3152ce9d8658724b228a` | `best_ep11.pth` → `models/external_vudang449_handwriting.pth` | Apache-2.0 |
| [DungHugging/vietocr-handwritten-finetune](https://huggingface.co/DungHugging/vietocr-handwritten-finetune) | `0827b42e17cb0e1c1bbda5aac11f7cc5c816e0ca` | `transformerocr.pth` → `models/external_dunghugging_handwriting.pth` | MIT |

Mỗi nguồn có `models/external_*_provenance.json` ghi URL model card, revision, tên tệp và SHA-256 đã kiểm tra. Script chỉ tải trọng số/config/vocab; không gửi ảnh phiếu lên dịch vụ ngoài. Không sử dụng số liệu quảng bá của model card làm kết quả bài tập. Quan hệ trùng lặp giữa dữ liệu huấn luyện bên ngoài và các phiếu hiện tại chưa được xác minh.

Trên validation cục bộ, chưa thích nghi tiếp, nguồn vudang449 cho tên CER 6,7%, so với 19,1% của v2; cả hai mới đúng toàn tên 1/7. Nguồn này đọc gần đúng hơn ở mức ký tự nhưng còn thừa dấu chấm/khác chữ, nên không đổi trực tiếp sang nó. Nguồn DungHugging khi dùng nguyên trạng bị thừa nhiều dấu chấm, chưa phù hợp phiếu hiện tại. Quan sát này là cơ sở thử fine-tune từ trọng số chữ viết tay, không phải kết quả cuối đã triển khai.

## Hai nhánh thích nghi đã hoàn tất

Hai nhánh hiện fine-tune transformer, khóa CNN; augmentation chỉ áp dụng train/replay. Dùng seed 2026, smoothing 0,05, xác suất ưu tiên ảnh sạch 0,65 và ba view cho mỗi crop train. LR của nhánh vudang449 là `2e-5`; nhánh DungHugging là `3e-5`.

Nhánh vudang449 dừng sau 14 epoch vì chỉ số tổng không cải thiện trong 6 epoch; checkpoint tổng tốt nhất là epoch 8, validation 18/28, CER 4,39%. Nhờ lưu riêng từng trường, tên ở epoch 14 vẫn được giữ: 4/7, CER 3,37%. Nhánh DungHugging chạy đủ 24 epoch, chọn epoch 24 theo chỉ số tổng: 18/28, CER 12,28%. Cả hai checkpoint tổng giữ replay 8/8.

Một lượt cuối chỉ train họ tên từ expert epoch 14, LR `1e-5`, dừng sau 6 epoch không cải thiện. Hai epoch đầu bằng mốc, bốn epoch sau kém hơn; không chọn trọng số mới. Dữ liệu glyph ghép không được dùng trong bất kỳ checkpoint triển khai nào.

Bundle đang dùng được chọn bằng validation, trước khi chạy test:

| Trường | Checkpoint | Epoch | Valid OCR thô | CER valid |
| --- | --- | ---: | ---: | ---: |
| Họ tên | `extended1_v4_handwriting_adapt_ho_ten.pth` | 14 | 4/7 | 3,37% |
| Ngày sinh | `extended1_v4_handwriting_adapt_ngay_sinh.pth` | 6 | 6/7 | 2,86% |
| Dân tộc | `extended1_v2_continued.pth` | Giữ v2 | 6/7 | 3,85% |
| Tôn giáo | `extended1_v4_dung_adapt.pth` | 24 | 6/7 | 6,98% |

Đánh giá lại ứng viên qua đúng luồng inference cho validation: **22/28 OCR thô, CER 3,95%; sau định dạng/giải mã 23/28**. Mô hình gốc và v3 được chạy lại trên chính các pixel này để đối chiếu. Sau đó kiểm tra bằng hai ảnh phiếu đầy đủ xác nhận replay đúng 8/8 cả OCR thô và sau giải mã; đây vẫn là dữ liệu đã train.

Trainer mới có `--field-checkpoints`, lưu epoch phù hợp riêng cho từng trường. `select_v4_bundle.py` chỉ xét checkpoint từ những lượt fine-tune cục bộ đã có report; không chọn trọng số công khai chưa thích nghi. Ứng viên phải có đúng nhãn, nhóm phiếu và pixel crop validation; hai crop replay của trường phải giữ đúng. CER validation phải tốt hơn, số trường đúng không được giảm so với v3. Vì epoch tốt nhất cho họ tên và ngày sinh có thể khác nhau, việc lưu riêng tránh mất một trường tốt khi tối ưu trường khác.

Trước khi thay bundle, `promote_v4_bundle.py` kiểm tra lại checksum, cùng tập nhãn/crop/cấu hình giải mã, các chỉ số tổng và từng trường trên validation/test cũ, rồi chạy lại hai phiếu replay bằng crop thực. Test cũ chỉ là bước kiểm tra chống suy giảm khi đưa vào demo, không được mô tả là đánh giá độc lập. Hai phiếu replay phải đúng 8/8 sau giải mã; chúng vẫn là dữ liệu train.

## Tái hiện quy trình

Chạy tại thư mục `Demo` sau khi đã có môi trường `.venv`, dữ liệu/nhãn/crop v2 và checkpoint v3. Cần giữ bản bundle v3 ở `models/extended1_v3_previous_bundle.json` trước khi thay model; không ghi đè bản này bằng bundle v4. Các lệnh dưới đây chạy hai nhánh tuần tự trên cùng dữ liệu; khi chạy song song phải dùng cache/thư mục dữ liệu riêng có pixel crop và nhãn giống nhau.

```powershell
& '.\.venv\Scripts\python.exe' '.\v4_handwriting_bases.py' --fetch-only

& '.\.venv\Scripts\python.exe' '.\finetune_v2.py' --data-dir '.\prepared\extended1_v2' --initial-weights '.\models\external_vudang449_handwriting.pth' --output '.\models\extended1_v4_handwriting_adapt.pth' --field-checkpoints --epochs 24 --patience 6 --threads 2 --lr 0.00002 --smoothing 0.05 --clean-probability 0.65
& '.\.venv\Scripts\python.exe' '.\finetune_v2.py' --data-dir '.\prepared\extended1_v2' --initial-weights '.\models\external_dunghugging_handwriting.pth' --output '.\models\extended1_v4_dung_adapt.pth' --field-checkpoints --epochs 24 --patience 6 --threads 2 --lr 0.00003 --smoothing 0.05 --clean-probability 0.65

& '.\.venv\Scripts\python.exe' '.\select_v4_bundle.py'
& '.\.venv\Scripts\python.exe' '.\evaluate_test.py' --data-dir '.\prepared\extended1_v2' --weights '.\models\extended1_v4_bundle_candidate.json' --output '.\models\extended1_v4_final_valid.json' --split valid --structured --previous-weights '.\models\extended1_v3_previous_bundle.json' --previous-label 'Fine-tune v3'
& '.\.venv\Scripts\python.exe' '.\evaluate_test.py' --data-dir '.\prepared\extended1_v2' --weights '.\models\extended1_v4_bundle_candidate.json' --output '.\models\extended1_v4_final_test.json' --split test --structured --previous-weights '.\models\extended1_v3_previous_bundle.json' --previous-label 'Fine-tune v3'
& '.\.venv\Scripts\python.exe' '.\promote_v4_bundle.py' --check-only
& '.\.venv\Scripts\python.exe' '.\promote_v4_bundle.py'

& '.\.venv\Scripts\python.exe' -m unittest test_extended1 -v
.\start_extended1.ps1
& '.\.venv\Scripts\python.exe' '.\smoke_demo.py'
```

`--check-only` lưu bằng chứng và chạy các bước kiểm tra, chưa thay bundle đang dùng. Sau khi bundle được thay, cần khởi động lại demo để nạp trọng số mới. Script chạy demo vẫn dùng cổng 7861 và đọc `models/extended1_bundle.json`; checkpoint v2 cũ còn được giữ riêng.

## Bằng chứng và chia sẻ model

Báo cáo máy chạy lưu ở `models/extended1_v4_final_valid.json`, `models/extended1_v4_final_test.json`; bundle ứng viên `models/extended1_v4_bundle_candidate.json`, SHA-256 `611c278c2b0642fe68fa959ecca218d687b537050b536a2bb0eb9c302e763839`. Trang đối chiếu từng trường được tạo lại ở `prepared/extended1_v2/results.html`. JSON và ảnh chứa dữ liệu cá nhân không đưa lên GitHub.

Khi chia sẻ bản v4 sang máy khác, nhận riêng `models/extended1_bundle.json` cùng **bốn** checkpoint trong bảng lựa chọn ở trên, mỗi file khoảng 145 MiB. Chỉ tải repo code chưa có trọng số fine-tune. V3 được giữ ở `models/extended1_v3_previous_bundle.json`; để quay lại, chạy demo với `--compare-weights models/extended1_v3_previous_bundle.json --default-model fine_tuned --port 7861` sau khi dừng phiên cũ.

Đã qua chín kiểm tra tự động và toàn bộ luồng Gradio: upload, crop tự động, đổi bản crop/lọc chấm, chặn OCR khi chưa xác nhận, sửa crop tay, hủy kết quả cũ, chọn model, xuất JSON và kiểm tra checksum theo trường. Hai phiếu mẫu cũ giữ đúng 8/8. Với hai ảnh đen/xanh 1 và 50, kiểm thử chức năng chạy thành công nhưng v4 sai tên như mô tả phía trên; không tính một lượt API thành công là một phiếu được đọc đúng.

Demo đang chạy tại <http://127.0.0.1:7861/>. Giao diện có ba lựa chọn khi đủ file: **VietOCR gốc**, **VietOCR Extended 1 v4**, **Fine-tune v3 · đối chiếu**. Chọn bản khác rồi bấm OCR lại để so trên cùng crop. Đã kiểm tra v3 vẫn trả lại các kết quả trước của phiếu 1 và 50; JSON ghi đúng model thực sự đã chạy. V3 không xuất hiện nếu thiếu checkpoint lịch sử.

Bằng chứng triển khai: `models/extended1_v4_bundle_evaluation.json`, `models/v4_replay_candidate.json`, `models/smoke_latest.json`. Bundle đã được kiểm tra không giảm OCR thô/CER/giải mã so với v3 ở cả tổng và từng loại trường trên valid/test cũ, rồi mới được thay vào `models/extended1_bundle.json`. Điều đó không bảo đảm từng phiếu riêng lẻ không giảm.
