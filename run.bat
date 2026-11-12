@echo off
echo Starting OCR App...
call conda activate ocrenv2
start "" http://127.0.0.1:5000/
python minor_project.py
pause
