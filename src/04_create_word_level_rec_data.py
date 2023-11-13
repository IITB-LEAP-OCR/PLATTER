import os
import pandas as pd
import cv2
import re
import json

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']

languages = ['bengali']

# for lang in languages:
   
img_dir = '/data/BADRI/OCR/data/CHIPS_1/test/images/'
txt_dir = '/data/BADRI/OCR/data/CHIPS_1/test/txt/'
output_dir = '/data/BADRI/OCR/intermediate/gt_CHIPS1/'

value = 0
lang = languages[value]
data = {}

if not os.path.exists(output_dir + lang + '/images/'):
    os.makedirs(output_dir + lang + '/images/')

for file in sorted(os.listdir(img_dir), key=natural_sort_key):

        
    df = pd.read_csv(txt_dir + file[:-4] + '.txt', sep=' ', header=None, names=['label', 'x1', 'y1', 'x2', 'y2'])
    df['id'] = df.index + 1
    img = cv2.imread(img_dir + file)
    
    height, width, _ = img.shape
    
    for _, row in df.iterrows():
        
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        # x1, y1, x2, y2 = int(row['x1']*width), int(row['y1']*height), int(row['x2']*width), int(row['y2']*height)
        cropped_image = img[y1:y2, x1:x2]
        name = file[:-4] +  '_' + str(int(row['id'])) + '.jpg'
        cv2.imwrite(output_dir + lang +  '/images/' +  name , cropped_image)
        data[name] = row['label']
        
    if(height!=1024):
        value+=1

        with open(output_dir + lang + '/labels.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)
            
        lang = languages[value]
        data = {}
        
        if not os.path.exists(output_dir + lang + '/images/'):
            os.makedirs(output_dir + lang + '/images/')
            
        