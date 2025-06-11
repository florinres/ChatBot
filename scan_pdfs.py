import os
from pdf2image import convert_from_path
import pytesseract

# Setare cale către Tesseract pe Windows (dacă e nevoie)
pytesseract.pytesseract.tesseract_cmd = r'D:\tess\tesseract.exe'

input_folder = "input"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, filename)
        print(f"[INFO] Procesăm: {filename}...")

        # Convertim fiecare pagină în imagine
        try:
            images = convert_from_path(pdf_path)
        except Exception as e:
            print(f"[EROARE] Nu pot converti: {filename} -> {e}")
            continue

        full_text = ""
        for idx, image in enumerate(images):
            try:
                text = pytesseract.image_to_string(image, lang='ron+eng')
                full_text += f"--- Pagina {idx+1} ---\n{text}\n"
            except Exception as e:
                print(f"[EROARE OCR] Pagina {idx+1}: {e}")

        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_folder, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"[OK] OCR finalizat: {output_filename}")
