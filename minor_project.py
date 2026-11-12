# minor_project.py
import os
import torch
import base64
import time
import io
import cv2
import numpy as np
from flask import Flask, request, render_template_string, send_file
from PIL import Image
import pytesseract

# TrOCR
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# EasyOCR
import easyocr


# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff"}

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_LANG = "eng"

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Lazy loaders
trocr_model = None
trocr_processor = None

trocr_handwritten_model = None
trocr_handwritten_processor = None

easy_reader = None


# ---------------- Helpers ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def read_image_fileobj(f):
    data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_image_from_datauri(datauri):
    base64_str = datauri.split(",", 1)[1]
    data = base64.b64decode(base64_str)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def cv2_to_datauri(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


# ---------------- Preprocessing ----------------
def preprocess_for_tesseract(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.5, (8, 8))
    gray = clahe.apply(gray)
    proc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 9)
    return proc


def prepare_for_trocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3, (8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------- OCR Engines ----------------
def load_trocr():
    global trocr_model, trocr_processor

    if trocr_model is None:
        model_name = "microsoft/trocr-base-printed"

        trocr_processor = TrOCRProcessor.from_pretrained(model_name)

        trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)

        trocr_model.eval()

    return trocr_model, trocr_processor


def load_trocr_handwritten():
    global trocr_handwritten_model, trocr_handwritten_processor

    if trocr_handwritten_model is None:
        model_name = "microsoft/trocr-base-handwritten"

        trocr_handwritten_processor = TrOCRProcessor.from_pretrained(model_name)

        trocr_handwritten_model = VisionEncoderDecoderModel.from_pretrained(model_name)

        trocr_handwritten_model.eval()

    return trocr_handwritten_model, trocr_handwritten_processor

def run_trocr(img_bgr):

    model, processor = load_trocr()

    # Convert OpenCV image to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    pil = Image.fromarray(img_rgb)

    # Process image
    pixel_values = processor(
        images=pil,
        return_tensors="pt"
    ).pixel_values

    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4
        )

    # Decode generated text
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text.strip()

def run_trocr_batch(crops, batch_size=4):

    model, processor = load_trocr()

    results = []

    for i in range(0, len(crops), batch_size):

        batch = crops[i:i + batch_size]

        # Convert OpenCV images to RGB PIL images
        pil_images = []

        for crop in batch:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        # Process multiple images together
        pixel_values = processor(
            images=pil_images,
            return_tensors="pt"
        ).pixel_values

        # Generate predictions
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=1
            )

        # Decode predictions
        batch_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        results.extend(batch_text)

    return [text.strip() for text in results]


def run_trocr_handwritten_batch(crops, batch_size=4):
    model, processor = load_trocr_handwritten()
    results = []

    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        pil_images = []

        for crop in batch:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        pixel_values = processor(
            images=pil_images,
            return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=1
            )

        batch_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        results.extend(batch_text)

    return [text.strip() for text in results]


def run_trocr_document(img_bgr):

    # Detect text regions
    boxes = detect_text_lines(img_bgr)

    # Remove overlapping detections
    boxes = remove_duplicate_boxes(boxes)

    # Sort regions by vertical centre, then horizontal position
    boxes.sort(
        key=lambda b: (
            (int(b[2]) + int(b[3])) / 2,
            int(b[0])
        )
    )

    crops = []

    for box in boxes:

        x1, x2, y1, y2 = map(int, box)

        # Add padding
        padding = 5

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)

        x2 = min(img_bgr.shape[1], x2 + padding)
        y2 = min(img_bgr.shape[0], y2 + padding)

        # Crop text region
        crop = img_bgr[y1:y2, x1:x2]

        if crop.size > 0:
            crops.append(crop)

    if not crops:
        return ""

    # Recognise all crops in batches
    extracted_lines = run_trocr_batch(crops, batch_size=4)

    # Combine recognised text
    final_text = "\n".join(
        text for text in extracted_lines if text.strip()
    )

    final_text = correct_ocr_text(final_text)

    return final_text


def run_trocr_handwritten_document(img_bgr):
    boxes = detect_text_lines(img_bgr)
    boxes = remove_duplicate_boxes(boxes)

    boxes.sort(
        key=lambda b: (
            (int(b[2]) + int(b[3])) / 2,
            int(b[0])
        )
    )

    crops = []

    for box in boxes:
        x1, x2, y1, y2 = map(int, box)

        padding = 5

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_bgr.shape[1], x2 + padding)
        y2 = min(img_bgr.shape[0], y2 + padding)

        crop = img_bgr[y1:y2, x1:x2]

        if crop.size > 0:
            crops.append(crop)

    extracted_lines = run_trocr_handwritten_batch(
        crops,
        batch_size=4
    )

    final_text = "\n".join(
        text for text in extracted_lines if text.strip()
    )

    final_text = correct_ocr_text(final_text)

    return final_text


def correct_ocr_text(text):
    corrections = {
        "ENGINERING": "ENGINEERING",
        "WISION": "VISION",
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    return text


def load_easyocr():
    global easy_reader
    if easy_reader is None:
        easy_reader = easyocr.Reader(["en"], gpu=False)
    return easy_reader

def detect_text_lines(img_bgr):
    reader = load_easyocr()

    horizontal_list, free_list = reader.detect(
        img_bgr,
        min_size=10,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4
    )

    boxes = horizontal_list[0] if horizontal_list else []

    # Sort boxes from top to bottom
    boxes = sorted(boxes, key=lambda box: box[2])

    grouped_lines = []

    for box in boxes:
        x1, x2, y1, y2 = map(int, box)
        box_center = (y1 + y2) / 2

        matched = False

        for line in grouped_lines:
            line_center = line["center"]

            if abs(box_center - line_center) < 15:
                line["boxes"].append([x1, x2, y1, y2])

                # Update average vertical center
                centers = [(b[2] + b[3]) / 2 for b in line["boxes"]]
                line["center"] = sum(centers) / len(centers)

                matched = True
                break

        if not matched:
            grouped_lines.append({
                "center": box_center,
                "boxes": [[x1, x2, y1, y2]]
            })

    # Merge boxes belonging to the same line
    merged_boxes = []

    for line in grouped_lines:
        line_boxes = line["boxes"]

        merged_boxes.append([
            min(b[0] for b in line_boxes),
            max(b[1] for b in line_boxes),
            min(b[2] for b in line_boxes),
            max(b[3] for b in line_boxes)
        ])

    return sorted(merged_boxes, key=lambda box: box[2])

def remove_duplicate_boxes(boxes, iou_threshold=0.5):

    if not boxes:
        return []

    boxes = [list(map(int, box)) for box in boxes]

    # Sort by bounding-box area, largest first
    boxes.sort(
        key=lambda b: (b[1] - b[0]) * (b[3] - b[2]),
        reverse=True
    )

    kept = []

    for box in boxes:
        x1, x2, y1, y2 = box

        duplicate = False

        for existing in kept:
            ex1, ex2, ey1, ey2 = existing

            # Calculate intersection
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (ex2 - ex1) * (ey2 - ey1)

            union = area1 + area2 - intersection

            iou = intersection / union if union > 0 else 0

            if iou > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(box)

    return sorted(kept, key=lambda b: (b[2], b[0]))


def run_easyocr(img_bin):
    reader = load_easyocr()
    res = reader.readtext(img_bin, detail=0)
    return "\n".join(res) if res else ""


def run_tesseract(img_bin):
    pil = Image.fromarray(img_bin)
    return pytesseract.image_to_string(pil, lang=OCR_LANG)


# ---------------- UI Template ----------------
TEMPLATE = """
<!doctype html>
<html>
<head>
<title>OCR Studio</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { background:#0f172a; color:white; font-family:Arial; padding:20px; }
.card { padding:20px; background:#16213e; border-radius:12px; }
img { max-width:100%; border-radius:10px; }
.pills button { padding:8px 14px; border-radius:20px; border:none; cursor:pointer; }
.pills .active { background:#06b6d4; color:black; }
.btn { padding:10px 14px; border-radius:6px; cursor:pointer; background:#06b6d4; border:none; color:black; }
.preview-box { background:#1b253b; padding:10px; border-radius:10px; }
</style>
</head>

<body>
<div class="card">

<h2>OCR Studio</h2>

<form id="mainForm" method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="image" required>
    <br><br>

    <div class="pills">
        <button type="button" class="pill active" data-engine="trocr">TrOCR Printed</button>
        <button type="button" class="pill" data-engine="trocr_handwritten">TrOCR Handwritten</button>
        <button type="button" class="pill" data-engine="easyocr">EasyOCR</button>
        <button type="button" class="pill" data-engine="tesseract">Tesseract</button>
    </div>

    <input type="hidden" name="engine" id="engineInput" value="trocr">
    <br>
    <button class="btn" type="submit">Upload & Extract</button>
</form>

<br><hr><br>

{% if original %}
<h3>Original</h3>
<div class="preview-box">
  <img src="{{ original }}">
</div>

<h3>Processed</h3>
<div class="preview-box">
  <img src="{{ processed }}">
</div>

<h3>Extracted Text</h3>
<div class="preview-box">
<pre style="white-space:pre-wrap;">{{ text }}</pre>
<div class="processing-time">
    Processing Time: {{ processing_time }} seconds
</div>
</div>
{% endif %}

</div>

<script>
const pills = document.querySelectorAll('.pill');
const engineInput = document.getElementById('engineInput');

pills.forEach(p => {
  p.onclick = () => {
    pills.forEach(x => x.classList.remove('active'));
    p.classList.add('active');
    engineInput.value = p.dataset.engine;
  };
});
</script>

</body>
</html>
"""


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload():
    orig_data = request.form.get("orig_data")
    engine = request.form.get("engine", "trocr")

    # CASE 1: Re-run fetch (small fragment)
    if orig_data:
        img = read_image_from_datauri(orig_data)
    else:
        # Normal file upload
        f = request.files["image"]
        img = read_image_fileobj(f)

    # Prep original preview
    original_uri = cv2_to_datauri(img)

    # Run engine
    start_time = time.time()
    if engine == "trocr":
        processed_uri = cv2_to_datauri(img)
        text = run_trocr_document(img)

    elif engine == "trocr_handwritten":
        processed_uri = cv2_to_datauri(img)
        text = run_trocr_handwritten_document(img)

    elif engine == "easyocr":
        proc = preprocess_for_tesseract(img)
        processed_uri = cv2_to_datauri(proc)
        text = run_easyocr(proc)

    else:
        proc = preprocess_for_tesseract(img)
        processed_uri = cv2_to_datauri(proc)
        text = run_tesseract(proc)

    processing_time = round(time.time() - start_time, 2)

    if not text.strip():
        text = "[No text extracted]"

    # If re-run → return small HTML fragment
    if orig_data:
        return f"""
        <!doctype html><html><body>
        <img id="origImg" src="{original_uri}">
        <img id="procImg" src="{processed_uri}">
        <pre id="textBox">{text}</pre>
        </body></html>
        """

    # Normal upload → return full UI with results inserted
    return render_template_string(
        TEMPLATE,
        original=original_uri,
        processed=processed_uri,
        text=text,
        processing_time=processing_time
    )


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)
