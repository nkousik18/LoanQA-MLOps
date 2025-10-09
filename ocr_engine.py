import pytesseract
import easyocr
import cv2

def ocr_tesseract(image):
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(image, config=custom_config)

def ocr_easyocr(img_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(img_path, detail=0)
    return "\n".join(results)
