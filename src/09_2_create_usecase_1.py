from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# Specify the font and size
font_size = 40
font_languages = {
    'bengali': '/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf',
    'gujarati': '/usr/share/fonts/truetype/lohit-gujarati/Lohit-Gujarati.ttf',
    'gurumukhi': '/usr/share/fonts/truetype/lohit-punjabi/Lohit-Gurmukhi.ttf',
    'hindi': '/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf',
    'kannada': '/usr/share/fonts/truetype/lohit-kannada/Lohit-Kannada.ttf',
    'malayalam': '/usr/share/fonts/truetype/lohit-malayalam/Lohit-Malayalam.ttf',
    'odia': '/usr/share/fonts/truetype/lohit-oriya/Lohit-Odia.ttf',
    'tamil': '/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf',
    'telugu': '/usr/share/fonts/truetype/lohit-telugu/Lohit-Telugu.ttf',
    'urdu': '/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf'
}
 # You may need to adjust the font path


model = 'vitstr_small'
folder_name = 'finetuned_CHIPS_1'

DATA_DIR = f'/data/BADRI/OCR/results/ocr/{folder_name}/{model}/'
DATA_DIR = '/data/BADRI/OCR/results/ocr/tesseract_CHIPS_1/'
output_dir = f'/data/BADRI/OCR/results/visual/{folder_name}/{model}/'
output_dir = '/data/BADRI/OCR/results/visual/tesseract_CHIPS_1/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for file in sorted(os.listdir(DATA_DIR), key=natural_sort_key):
    
    # Create a blank image with white background
    width, height = 1024, 1024
    background_color = "white"
    img = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(img)

    
    language = file.split('_')[0]
    df = pd.read_csv(DATA_DIR + file, sep=' ', names=['label', 'x1', 'y1', 'x2', 'y2'])
    data = df.values.tolist()
    try:
        font = ImageFont.truetype(font_languages[language], font_size) 
        for text, x1, y1, x2, y2 in data:
            draw.text((x1, y1), text, font=font, fill="black")
    except:
        print(language)
        print(file)

    

    # Save the synthetic page
    img.save(output_dir + file[:-4] + '.jpg')
