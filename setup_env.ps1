# TechcodeX local environment setup (Windows / PowerShell)
# Installs UI + tokenizer/dataset dependencies, then installs torch-directml
# so TechcodeX runs natively on your Windows AMD GPU.

Write-Host "=== TechcodeX Environment Setup ===" -ForegroundColor Cyan

Write-Host "`nInstalling core Python dependencies (transformers, datasets, gradio, pymupdf, pytesseract, Pillow, langdetect, huggingface_hub)..." -ForegroundColor Yellow
pip3 install transformers datasets gradio pandas pymupdf pytesseract Pillow langdetect huggingface_hub

Write-Host "`n=== OCR support for image (PNG/JPG) datasets ===" -ForegroundColor Cyan
Write-Host "pytesseract only installs the Python wrapper. To OCR images in Tab 0, you also need" -ForegroundColor Yellow
Write-Host "the Tesseract OCR engine itself installed separately and on PATH:"
Write-Host "     https://github.com/UB-Mannheim/tesseract/wiki"
Write-Host "PDFs with a real text layer do NOT need this — only standalone image files do."

Write-Host "`n=== PyTorch + AMD GPU (DirectML) binding ===" -ForegroundColor Cyan
Write-Host "Installing base PyTorch from the official index:" -ForegroundColor Green
Write-Host "     pip3 install torch --index-url https://pytorch.org"
pip3 install torch --index-url https://pytorch.org

Write-Host "`nInstalling torch-directml to bind PyTorch to your native Windows AMD GPU:" -ForegroundColor Green
Write-Host "     pip3 install torch-directml"
pip3 install torch-directml

Write-Host "`nAfter installing, verify DirectML sees your AMD GPU with:" -ForegroundColor Cyan
Write-Host "     python -c ""import torch_directml; print(torch_directml.device())"""
Write-Host ""
Write-Host "=== Setup complete. Launch the dashboard with: python app_ui.py ===" -ForegroundColor Cyan
