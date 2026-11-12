# Handwritten and Printed Text Recognition Using Multi-Engine OCR

A Python-based Optical Character Recognition (OCR) application that extracts text from images and documents using multiple OCR engines.

The application integrates Tesseract, EasyOCR, and two TrOCR models to recognise printed, handwritten, and stylised text through a Flask-based web interface.

## Features

- **Multiple OCR Engines:** Supports Tesseract, EasyOCR, TrOCR Printed, and TrOCR Handwritten.
- **Printed Text Recognition:** Extracts text from printed documents and images.
- **Handwritten Text Recognition:** Uses a handwriting-specific TrOCR model.
- **Text Detection:** Uses EasyOCR to detect text regions in document images.
- **Image Preprocessing:** Applies OpenCV techniques to improve image quality for OCR.
- **Multi-line Recognition:** Detects and processes multiple text regions.
- **Batch Inference:** Processes multiple text regions in batches using TrOCR.
- **Text Post-processing:** Applies basic corrections to recognised text.
- **Performance Measurement:** Displays OCR processing time.
- **Web Interface:** Provides an interactive interface built with Flask.

## Technologies Used

- Python
- Flask
- OpenCV
- NumPy
- Tesseract OCR
- EasyOCR
- Hugging Face Transformers
- TrOCR
- PyTorch
- Pillow

## OCR Engines

| OCR Engine | Purpose |
|---|---|
| Tesseract | Printed text recognition |
| EasyOCR | Text detection and recognition |
| TrOCR Printed | Printed text recognition using transformer models |
| TrOCR Handwritten | Handwritten text recognition using transformer models |

## Project Structure

```text
m_pro/
│
├── minor_project.py
├── requirements.txt
├── README.md
├── run.bat
│
├── uploads/
│
└── venv/
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd m_pro
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Tesseract OCR

Install the Tesseract OCR application separately.

Make sure the executable path is correctly configured in `minor_project.py`.

Example:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 6. Run the Application

```bash
python minor_project.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Performance Evaluation

The application measures the processing time of each OCR engine.

Initial measurements on the development laptop using one test image:

| OCR Engine | Processing Time |
|---|---:|
| Tesseract | 0.25 seconds |
| EasyOCR | 2.45 seconds |
| TrOCR | 86 seconds |

*Note: These are preliminary measurements from a single test image, not benchmark results. Actual performance depends on image complexity, hardware, model loading and other factors.*

## Future Improvements

- Improve OCR accuracy on complex handwriting.
- Optimise TrOCR inference speed.
- Improve text-region detection and reading order.
- Add support for additional languages.
- Expand testing across different document types.
- Improve the user interface and document export options.

## Project Objective

The objective of this project is to develop and evaluate a multi-engine OCR system capable of recognising printed and handwritten text from images, while comparing the performance and capabilities of different OCR approaches.

---

**Developed as an academic project using Python and Machine Learning.**