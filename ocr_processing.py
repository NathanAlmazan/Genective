from tempfile import TemporaryDirectory
from pathlib import Path
 
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

image_file_list = []

text_file = "E:\\BSCS 4-3\\Thesises\\genective\\corpus\\corpus_5.txt"
pdf_file = "E:\\BSCS 4-3\\Thesises\\genective\\documents\\documentD.pdf"

with TemporaryDirectory() as tempdir:
    pdf_pages = convert_from_path(pdf_file, 500, poppler_path="C:\\Program Files\\poppler-23.11.0\\Library\\bin")
    
    for page_enumeration, page in enumerate(pdf_pages, start=1):
        # Create a file name to store the image
        filename = f"{tempdir}\page_{page_enumeration:03}.jpg"

        # Save the image of the page in system
        page.save(filename, "JPEG")
        image_file_list.append(filename)
        
    with open(text_file, "a") as output_file:
        for image_file in image_file_list:
            text = str(((pytesseract.image_to_string(Image.open(image_file)))))
            text = text.replace("-\n", "")
            output_file.write(text)
        