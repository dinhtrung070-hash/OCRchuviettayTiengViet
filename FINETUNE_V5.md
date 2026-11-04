# Thử nghiệm cải thiện VietOCR · v5 · 04/11/2026

## Kết quả đối chiếu cuối

**Đã triển khai và kiểm thử bản v5.** Bundle được đóng băng ở `models/extended1_v5_bundle_candidate.json`, vượt các kiểm tra chống suy giảm và được sao sang `models/extended1_bundle.json`. Các số bên dưới lấy từ `models/extended1_v5_final_valid.json` và `models/extended1_v5_final_test.json`; kiểm tra replay/giao diện ghi ở cuối báo cáo. Không dùng số liệu train để thay cho kết quả test.

V5 **không có lượt huấn luyện mới nào vượt được mốc để chọn trọng số mới**. Đã thử lựa chọn lại checkpoint dân tộc, nhưng phương án đó bị loại ở bước kiểm tra chống suy giảm. Ứng viên hiện giữ **toàn bộ trọng số v4**, chỉ thay cách đề xuất theo danh mục cho dân tộc/tôn giáo. Phần chuẩn bị dữ liệu và nguồn khởi tạo bên ngoài xem [FINETUNE_V4.md](FINETUNE_V4.md).

| Chỉ số | V4 | V5 đang dùng |
| --- | ---: | ---: |
| VALID OCR thô | 22/28 | 22/28 |
| CER valid | 3,95% | 3,95% |
| Valid sau định dạng | 23/28 | 23/28 |
| **Valid sau giải mã theo trường** | 23/28 | **24/28** |
| TEST OCR thô | 13/28 | 13/28 |
| CER test | 13,81% | 13,81% |
| Test sau định dạng | 16/28 | 16/28 |
| **Test sau giải mã theo trường** | 19/28 | **21/28** |

Phần cải thiện đo được là **giải mã sau OCR**, không phải tăng độ chính xác mô hình thần kinh. Họ tên/ngày sinh vẫn có thể sai như v4. Các chỉ số sau giải mã không có sửa tay nhưng dùng danh mục/khuôn trường, phải được tách khỏi OCR thô.

| Trường trên test, sau định dạng/giải mã | V4 | V5 |
| --- | ---: | ---: |
| Họ tên | 3/7 | 3/7 |
| Ngày sinh | 4/7 | 4/7 |
| **Dân tộc** | 5/7 | **7/7** |
| Tôn giáo | 7/7 | 7/7 |

Theo nhánh phân loại mực tự động trên test: xanh có 16 trường, số đúng sau giải mã tăng 11→13; đen/tối có 12 trường, giữ 8/12. Phân loại mực này chưa có nhãn độc lập do nhóm xác nhận. Kết quả test vẫn chỉ là khảo sát trên bộ đã dùng lại, không được diễn giải thành bảo đảm cho các phiếu mới.

## Dữ liệu và nguyên tắc chọn phương án

Giữ 137 crop train, 28 crop validation, 28 crop test theo cách chia v2; tám crop của hai phiếu mẫu cũ là replay trong train. Các lượt thử mới chọn phương án bằng train/validation, không đưa ảnh hoặc crop con của valid/test vào train. Tập test cũ đã được xem nhiều lần trong dự án; bước đối chiếu cuối chỉ có vai trò khảo sát và kiểm tra chống suy giảm khi triển khai.

Chưa có ID người viết thật và nhãn mới được đọc từ ảnh, chưa được nhóm đối chiếu chéo đầy đủ. Bảy nhóm phiếu validation không chứng minh có bảy người viết độc lập. Không mô tả kết quả hiện tại là độ chính xác trên ảnh mới ngoài thực tế.

**Bộ test cũ đã ảnh hưởng đến bản triển khai:** ứng viên đổi checkpoint dân tộc được chọn từ train/valid, đóng băng rồi đánh giá; kết quả test kém đi khiến hệ thống quay về trọng số dân tộc v4. Vì vậy không được mô tả lựa chọn bản cuối là chỉ dựa vào validation hoặc coi bộ test này là kiểm định độc lập.

## Thử lại tiền xử lý trên model v4

Các thử nghiệm đầu vào trước đó chủ yếu dùng v3, vì vậy v5 đo lại trên chính bundle v4. Tất cả dùng cùng 28 crop validation; không có nhãn hay quy tắc riêng cho từng ảnh ở lúc suy luận.

| Đầu vào | OCR thô đúng | CER | Sau định dạng/giải mã |
| --- | ---: | ---: | ---: |
| Mốc v4, preprocessing v2 | 22/28 | 3,95% | 23/28 |
| Giữ cụm mực chính, bỏ mảnh xa theo chiều ngang | 22/28 | 3,95% | 23/28 |
| Thu vùng tên theo cụm chữ | 22/28 | 3,95% | 23/28 |
| Mặt nạ mực xanh, giữ cường độ nét gốc | 21/28 | 4,39% | 22/28 |
| Xóa thêm chấm rời trong khoảng cách chữ | 19/28 | 5,26% | 20/28 |
| Tỷ lệ ngang 0,93 | 18/28 | 6,14% | 19/28 |
| Tỷ lệ ngang 1,07 | 18/28 | 7,89% | 21/28 |
| Thêm đệm dọc 5% | 14/28 | 9,65% | 17/28 |

Không thay preprocessing đang dùng vì chưa có phương án tốt hơn trên validation. Vẫn giữ `stroke_preserving_v2`, xử lý được cả mực xanh và đen/tối, ưu tiên không xóa mất dấu hoặc nét dính vào dòng chấm. Những chấm còn lại có thể tiếp tục gây lỗi; ảnh trông sạch hơn không đủ để kết luận OCR tốt hơn. Báo cáo: `models/v5_input_valid.json`.

## Các lượt huấn luyện và trộn checkpoint bị loại

| Thử nghiệm | Dữ liệu/phạm vi | Kết quả validation | Quyết định |
| --- | --- | --- | --- |
| Fine-tune phần CNN cuối cùng với transformer cho họ tên | Khởi từ expert v4 epoch 14, 31 tên train + 2 replay, CNN LR bằng 0,1 LR transformer `1e-5` | Hai epoch đầu bằng 4/7, CER 3,37%; epoch 3–6 giảm còn 3/7, CER 4,49% | Dừng sau 6 epoch, không chọn checkpoint |
| Thêm ảnh tên thu gọn vùng chữ | 31 crop tên gốc + 31 view thu vùng từ train; valid giữ nguyên pixel; CNN khóa, LR `1e-5` | Sáu epoch đều 4/7, CER 3,37% | Không vượt mốc |
| Tăng cường bằng crop từ/âm tiết | 31 tên đầy đủ, lặp 3 lần; 87 crop từ đã được kiểm tra từ 30 phiếu train; CNN khóa, LR `5e-6` | Hai epoch đầu bằng 4/7; epoch 3–6 còn 3/7, CER 4,49%; replay giữ 2/2 | Không chọn trọng số mới |
| Fine-tune riêng dân tộc | Khởi từ expert v4, 36 crop train + 2 replay, CNN khóa, LR `2e-5` | Sáu epoch giữ 6/7, CER 3,85% | Không vượt mốc |
| Trộn trọng số hai checkpoint tên v4 | `(1-alpha) * tên epoch 14 + alpha * checkpoint tổng epoch 8`; alpha 0,25 / 0,5 / 0,75 | Cả ba chỉ 3/7, CER 4,49%, so với 4/7 và 3,37% của expert | Loại cả ba |

Các crop từ chỉ là tăng cường từ train, không phải 87 phiếu hay người viết mới. Đã kiểm tra từng nguồn thuộc annotation train và nhãn từ khớp vị trí âm tiết trong tên nguồn. Provenance ở `prepared/extended1_v5_words/word_provenance.json`. Lượt tăng cường từ làm mất một dấu mà mốc v4 đã đọc đúng trên validation, nên không đổi model chỉ vì train loss giảm.

Các báo cáo cục bộ: `models/extended1_v5_name_visual.json`, `extended1_v5_name_focus.json`, `extended1_v5_name_words.json`, `extended1_v5_ethnicity.json`, `extended1_v5_name_average.json`. Các lượt train đều không chọn checkpoint mới.

Đã thử lại beam search rộng 8 cho họ tên, ba mức chuẩn hóa độ dài và hai chế độ bảng chữ. Cả sáu cấu hình chỉ đạt 3/7, CER 4,49%, kém greedy 4/7, CER 3,37%. Giữ greedy; không dùng từ điển họ tên hoặc danh sách đáp án. Báo cáo: `models/v5_decode_valid.json`.

## Lựa chọn lại checkpoint dân tộc

Ba checkpoint cũ có cùng kết quả validation dân tộc: 6/7, CER 3,85%, sau giải mã 6/7; replay đều đúng 2/2. Thử nghiệm dùng độ khớp TRAIN để phá hòa, với quy tắc CER train thấp hơn, rồi số đúng train cao hơn.

| Checkpoint cũ | Epoch/nguồn | TRAIN OCR thô | CER train | VALID OCR thô |
| --- | --- | ---: | ---: | ---: |
| `extended1_v2_continued.pth` | Mốc dân tộc v4 | 27/36 | 11,11% | 6/7 |
| `extended1_v4_handwriting_adapt_dan_toc.pth` | V4, nhánh vudang449 | 30/36 | 8,33% | 6/7 |
| `extended1_v4_dung_adapt_dan_toc.pth` | **V4, nhánh DungHugging, epoch 19** | **32/36** | **8,33%** | **6/7** |

Ứng viên đầu tiên chọn dòng cuối làm checkpoint dân tộc. Đây là **thử nghiệm phá hòa dựa trên train**, không phải bằng chứng nó tổng quát tốt hơn. Validation không cải thiện; khi đánh giá trên test cũ, CER tổng tăng từ 13,8% lên khoảng 14,2%, kết quả sau giải mã giảm từ 19/28 xuống 18/28. Công cụ kiểm tra từ chối triển khai, và trọng số dân tộc được trả về `extended1_v2_continued.pth` như v4. Không gọi mức tăng train là mức tăng độ chính xác thực tế.

Nguồn so sánh train/valid: `models/v5_ethnicity_heads.json`. Ứng viên bị loại và bằng chứng được giữ riêng ở `models/v5_rejected_ethnicity_bundle.json`, `v5_rejected_ethnicity_valid.json`, `v5_rejected_ethnicity_test.json`. Các tệp cuối của ứng viên giữ trọng số v4 được đánh giá lại riêng, không ghi đè lịch sử thất bại.

## Đề xuất tôn giáo theo danh mục, có quyền từ chối

V5 thử cách chấm điểm các cách viết hoa/thường của cùng một mục danh mục bằng xác suất chuỗi có điều kiện theo ảnh. Các cách viết của một mục được gom về cùng đáp án chuẩn; không so hai cách viết hoa như hai tôn giáo khác nhau. Chỉ xét mục đủ gần OCR thô theo khoảng cách ký tự, không ép mọi nội dung về một danh mục.

Ngưỡng được chọn bằng TRAIN, sau đó kiểm tra VALID: score tối thiểu `0.35980080941540404`, biên điểm log tương ứng `odds = 2`. Score là hàm mũ của trung bình log-likelihood token; **không phải xác suất đáp án đúng đã được hiệu chỉnh**. Điều kiện biên cũng không chứng minh đáp án đúng hay độ tin cậy 2:1 ngoài thực tế.

Tập chọn ngưỡng chỉ có bảy đề xuất được chấp nhận, đúng 7/7 trên TRAIN. Mức precision 100% đó mang tính lạc quan vì chính train được dùng chọn ngưỡng, support rất nhỏ, và validation đã được khảo sát nhiều lần. Ban đầu có lượt thử cố định biên 3:1; sau đó thêm sàn score 0,25 và khảo sát các biên cố định 2/3/5. Quy tắc cuối chọn bằng precision/support train, không chọn bằng test.

Trên báo cáo kiểm tra hai trường danh mục bằng model v4, cách cũ đúng 12/14 trên validation; đề xuất mới đúng 13/14, với dân tộc giữ 6/7 và tôn giáo tăng 6/7 → 7/7. Sau khi bỏ nhánh đổi checkpoint dân tộc, ứng viên mới **bật chính sách cho cả dân tộc và tôn giáo**, trên đúng các trọng số v4 đã dùng chọn ngưỡng. Không chuyển nguyên ngưỡng sang model khác. Chỉ số của toàn bộ bundle cần lấy từ đánh giá cuối, không suy ra độ chính xác test từ 13/14 validation này.

Khi không đủ score hoặc biên giữa hai mục, hệ thống giữ kết quả hiện có. OCR thô được giữ nguyên trong JSON; nếu có đề xuất, JSON ghi nguồn đề xuất và giao diện yêu cầu đối chiếu ảnh. Người dùng vẫn sửa và xác nhận nội dung trước khi xuất. Chính sách chỉ được bật qua `category_policies` trong bundle v5; các bundle v4/v3 không có thiết lập đó giữ hành vi trước. Không thêm từ điển họ tên hoặc đổi cách giải mã ngày sinh.

Báo cáo và mã: `models/v5_category_valid.json`, `v5_category_trials.py`, `v5_category_decode.py`.

## Ứng viên và quy trình đối chiếu

| Trường | Checkpoint của ứng viên v5 |
| --- | --- |
| Họ tên | `extended1_v4_handwriting_adapt_ho_ten.pth` — giữ v4 epoch 14 |
| Ngày sinh | `extended1_v4_handwriting_adapt_ngay_sinh.pth` — giữ v4 epoch 6 |
| Dân tộc | `extended1_v2_continued.pth` — giữ nguyên trọng số đang dùng ở v4 |
| Tôn giáo | `extended1_v4_dung_adapt.pth` — giữ v4 epoch 24; thêm chính sách đề xuất |

`select_v5_bundle.py` kiểm tra nhãn, nhóm validation, pixel crop và replay khi xét các lượt train; kiểm tra checksum checkpoint khi xét độ khớp train của ba model dân tộc; đóng băng chính sách đề xuất trong bundle. Bản đầu có thử phá hòa dân tộc theo train. Sau khi phương án đó bị loại, chạy `--keep-v4-ethnicity` để giữ toàn bộ trọng số v4 và dùng chính sách trên đúng các model đã hiệu chỉnh.

`promote_v4_bundle.py` đã có tham số phiên bản để dùng lại cho v5. Công cụ phải đối chiếu ứng viên với snapshot v4 trên cùng annotation/cấu hình, kiểm tra không giảm OCR thô, CER hoặc kết quả sau giải mã ở tổng và từng trường, rồi kiểm tra hai phiếu replay bằng ảnh đầy đủ. Chỉ thay bundle đang dùng sau khi các bước này thành công. Test cũ ở bước này là kiểm tra chống suy giảm, không trở thành đánh giá độc lập.

## Tái hiện chọn bundle, đánh giá và chạy demo

Chạy tại thư mục `Demo` khi đã có dữ liệu/nhãn/crop v2 và các báo cáo thử nghiệm tương ứng. Cần giữ bundle v4 cũ đúng tên `models/extended1_v4_previous_bundle.json`; không ghi đè nó bằng v5. Các ảnh, nhãn, JSON dự đoán và trọng số được giữ ở máy train, không đưa lên repo công khai.

```powershell
& '.\.venv\Scripts\python.exe' '.\select_v5_bundle.py' --keep-v4-ethnicity
& '.\.venv\Scripts\python.exe' '.\evaluate_test.py' --data-dir '.\prepared\extended1_v2' --weights '.\models\extended1_v5_bundle_candidate.json' --output '.\models\extended1_v5_final_valid.json' --split valid --structured --previous-weights '.\models\extended1_v4_previous_bundle.json' --previous-label 'Fine-tune v4'
& '.\.venv\Scripts\python.exe' '.\evaluate_test.py' --data-dir '.\prepared\extended1_v2' --weights '.\models\extended1_v5_bundle_candidate.json' --output '.\models\extended1_v5_final_test.json' --split test --structured --previous-weights '.\models\extended1_v4_previous_bundle.json' --previous-label 'Fine-tune v4'
& '.\.venv\Scripts\python.exe' '.\promote_v4_bundle.py' --version 5 --previous-version 4 --check-only
& '.\.venv\Scripts\python.exe' '.\promote_v4_bundle.py' --version 5 --previous-version 4

& '.\.venv\Scripts\python.exe' -m unittest test_extended1 test_v5_category -v
.\start_extended1.ps1
& '.\.venv\Scripts\python.exe' '.\smoke_demo.py'
& '.\.venv\Scripts\python.exe' '.\make_review_page.py' --data-dir '.\prepared\extended1_v2' --source-dir '.\forms\dataset_20261102' --evaluation '.\models\extended1_v5_final_test.json'
```

`--check-only` chạy các kiểm tra và replay nhưng không đổi bundle đang dùng. Chỉ cần chạy bước triển khai khi đã kiểm tra xong; sau khi thay bundle cần khởi động lại demo để nạp checkpoint/chính sách đúng phiên bản. Demo tiếp tục dùng cổng 7861.

## Bằng chứng triển khai và kiểm thử

Bundle active có SHA-256 `81e8018dbcbb17e1f761ee505486391a3c49d29b7a7ac8895058c359edbf75e5`. Trước khi thay bundle, đã đối chiếu lại metric v4 thực chạy, annotation, cấu hình, checkpoint và không giảm raw/CER/giải mã ở tổng hoặc từng loại trường. Hai ảnh mẫu cũ giữ **8/8 OCR thô và sau giải mã**; đây là replay train, không phải test độc lập.

Đã qua **15 test tự động** và kiểm thử Gradio upload → crop → bật/tắt lọc chấm → xác nhận → OCR → sửa crop tay → hủy kết quả cũ → xuất JSON. Đã chạy với hai phiếu mẫu, phiếu mực đen và xanh; kiểm tra chính sách và checksum đúng theo từng trường. Một lượt gọi API thành công không đồng nghĩa đọc đúng toàn phiếu: lỗi tên cũ vẫn xuất hiện như v4.

Demo chạy ở <http://127.0.0.1:7861/>. Giao diện có lựa chọn model gốc, v5, v4 và v3 khi đủ file. Đã thử chuyển từ v5 sang v4/v3 trên cùng crop; JSON ghi đúng phiên bản và chính sách v5 không lan sang cấu hình model cũ. Người dùng cần bấm OCR lại sau khi chọn bản khác.

Bằng chứng cục bộ: `models/extended1_v5_final_valid.json`, `extended1_v5_final_test.json`, `extended1_v5_bundle_evaluation.json`, `v5_replay_candidate.json`, `smoke_latest.json`; bảng đối chiếu ảnh ở `prepared/extended1_v2/results.html`. V4 được giữ ở `models/extended1_v4_previous_bundle.json`. Khi chia sẻ model v5, cần bundle mới và cùng bốn `.pth` đã dùng cho v4; không có `.pth` v5 mới cần tải.

Không mô tả v5 là đã học được cách đọc tên/ngày sinh chính xác hơn nếu hai trường vẫn giữ cùng trọng số và cách giải mã. Những lỗi cũ có thể còn nguyên; điều hợp lý là ghi đúng phần cải thiện được đo và giữ cơ chế người dùng kiểm tra/sửa nội dung.
