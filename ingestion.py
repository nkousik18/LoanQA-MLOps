import os
from pdf2image import convert_from_path
import cv2
from PIL import Image

def pdf_to_images(pdf_path, output_dir="temp_pages"):
    os.makedirs(output_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=300, fmt="png", grayscale=True)
    img_paths = []
    for i, page in enumerate(pages):
        img_path = os.path.join(output_dir, f"page_{i+1}.png")
        page.save(img_path)
        img_paths.append(img_path)
    return img_paths

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3,3), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img
