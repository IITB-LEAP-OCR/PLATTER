import cv2
import os
import re
from tqdm import tqdm
from pytesseract import image_to_data, Output
import pandas as pd

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']

OCR_DATA_PATH = '/data/BADRI/OCR/data/PhDIndic11/images/'
output_dir = '/data/BADRI/OCR/results/ocr/tesseract_PhDIndic11/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# tesseract models for different languages
terms = {
    'bengali': 'ben',
    'gujarati': 'guj',
    'gurumukhi': 'pan',
    'hindi': 'hin',
    'kannada': 'kan',
    'malayalam': 'mal',
    'odia': 'ori',
    'tamil': 'tam',
    'telugu': 'tel',
    'urdu': 'urd'
}


for image_file in tqdm(sorted(os.listdir(OCR_DATA_PATH), key=natural_sort_key)):
    predictions = []
    language = image_file.split('_')[0]
    image = cv2.imread(OCR_DATA_PATH + image_file)
    d = image_to_data(image, output_type=Output.DICT, lang=terms[language])
    for i in range(len(d['level'])):
        if d['level'][i]==5:
            values = []
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            (x, y, w, h) = (int(x), int(y), int(w), int(h))
            values.append(d['text'][i])
            values.append(d['left'][i])
            values.append(d['top'][i])
            values.append(d['left'][i] + d['width'][i])
            values.append(d['top'][i] + d['height'][i])
            predictions.append(values)
    
    df = pd.DataFrame(predictions, columns=['pred', 'x1', 'y1', 'x2', 'y2'])
    df.to_csv(f'{output_dir}{image_file[:-4]}.txt', sep=' ', index=False, header=False)
    
    
    
# # For PhDIndic11
# value = 0
# language = languages[value]
# for image_file in tqdm(sorted(os.listdir(OCR_DATA_PATH), key=natural_sort_key)):
    
#     language = image_file.split('_')[0]
#     predictions = []
#     image = cv2.imread(OCR_DATA_PATH + image_file)
#     height, width, _ = image.shape
#     d = image_to_data(image, output_type=Output.DICT, lang=terms[language])
#     for i in range(len(d['level'])):
#         if d['level'][i]==5:
#             values = []
#             (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#             (x, y, w, h) = (int(x), int(y), int(w), int(h))
#             values.append(d['text'][i])
#             values.append(d['left'][i])
#             values.append(d['top'][i])
#             values.append(d['left'][i] + d['width'][i])
#             values.append(d['top'][i] + d['height'][i])
#             predictions.append(values)
    
#     df = pd.DataFrame(predictions, columns=['pred', 'x1', 'y1', 'x2', 'y2'])
#     df.to_csv(f'{output_dir}{image_file[:-4]}.txt', sep=' ', index=False, header=False)
    