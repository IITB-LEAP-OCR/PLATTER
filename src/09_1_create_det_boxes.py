import cv2
import os
import re
import pandas as pd

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def draw_bboxes(image_path, txt_path, output_path):
    df = pd.read_csv(txt_path, sep=' ', names=['label', 'x1', 'y1', 'x2', 'y2'])
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['x1']*width), int(row['y1']*height), int(row['x2']*width), int(row['y2']*height)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    cv2.imwrite(output_path, img)
    
    
    
folders = ['finetuned_CHIPS1', 'pretrained_CHIPS1']

for folder in folders:
    image_path = '/data/BADRI/OCR/data/CHIPS_1/test/images/'
    txt_path = f'/data/BADRI/OCR/results/detection/txt/{folder}/'
    output_path = f'/data/BADRI/OCR/results/detection/images/{folder}/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in sorted(os.listdir(image_path), key=natural_sort_key):
        draw_bboxes(image_path + file, txt_path +  file[:-4] + '.txt', output_path + file)
    