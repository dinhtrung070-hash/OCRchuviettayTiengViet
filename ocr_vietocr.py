import json
import numpy as np
from pathlib import Path

from PIL import Image, ImageEnhance
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


BASE_DIR = Path(__file__).parent


def crop_ratio(image, roi):
    width, height = image.size
    x1, y1, x2, y2 = roi

    crop = image.crop((
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height),
    ))

    # Phóng to vì ảnh mẫu khá nhỏ
    crop = crop.resize(
        (crop.width * 3, crop.height * 3),
        Image.Resampling.LANCZOS,
    )

    crop = ImageEnhance.Contrast(crop).enhance(1.5)
    return crop


template_path = BASE_DIR / "phieu_mau.png"
filled_path = BASE_DIR / "phieudaviet1.png"

if not template_path.exists():
    raise FileNotFoundError(f"Không tìm thấy: {template_path}")

if not filled_path.exists():
    raise FileNotFoundError(f"Không tìm thấy: {filled_path}")

filled_image = Image.open(filled_path).convert("RGB")

# Bỏ khoảng trắng phía trên và phía dưới ảnh
gray = np.array(filled_image.convert("L"))
mask = gray < 240
row_has_content = mask.any(axis=1)

ys = np.where(row_has_content)[0]
if len(ys) > 0:
    y1 = max(int(ys[0]) - 5, 0)
    y2 = min(int(ys[-1]) + 5, filled_image.height)
    filled_image = filled_image.crop((0, y1, filled_image.width, y2))

# Tọa độ tương đối theo ảnh phiếu hiện tại.
# Nếu crop chưa đúng, chỉnh các giá trị này.
ROIS = {
    "ho_ten": (0.23, 0.45, 0.45, 0.70),
    "ngay_sinh": (0.21, 0.72, 0.38, 0.98),
    "dan_toc": (0.53, 0.72, 0.62, 0.98),
    "ton_giao": (0.78, 0.72, 0.87, 0.98),
}
from PIL import ImageDraw

debug_image = filled_image.copy()
draw = ImageDraw.Draw(debug_image)

for field, roi in ROIS.items():
    width, height = debug_image.size
    x1, y1, x2, y2 = roi

    box = (
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height),
    )

    draw.rectangle(box, outline="red", width=2)
    draw.text((box[0], box[1]), field, fill="red")

debug_image.save(BASE_DIR / "debug_rois.png")
print("Đang tải model VietOCR...")

config = Cfg.load_config_from_name("vgg_transformer")
config["device"] = "cpu"

predictor = Predictor(config)

result = {}

for field, roi in ROIS.items():
    crop = crop_ratio(filled_image, roi)

    crop_path = BASE_DIR / f"crop_{field}.png"
    crop.save(crop_path)

    text = predictor.predict(crop)
    result[field] = " ".join(text.split())

output_path = BASE_DIR / "ocr_result_vietocr.json"
output_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nKẾT QUẢ OCR:")
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"\nĐã lưu kết quả tại: {output_path}")