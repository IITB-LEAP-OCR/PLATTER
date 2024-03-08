import os
import pandas as pd
import cv2
import re
import json

from config import DATA_DIR, RESULTS_DIR

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']

lang_range = {
    'bengali':0,
    'gujarati':1,
    'gurumukhi':2,
    'hindi':3,
    'kannada':4,
    'malayalam':5,
    'odia':6,
    'tamil':7,
    'telugu':8,
    'urdu':9
}

placeholders = ['আফগানিস্তান', 'આધાર-પુરાવાઓ', 'ਸਮਾਜੀ-ਸਭਿਆਚਾਰਕ', 'कालाबाजारी', 'ಚಿತ್ರಮಂದಿರಗಳನ್ನು', 'തിരികൊളുത്തിയിരിക്കുകയാണ്', 'ବାଙ୍ଗାଲୋରଠାରୁ', 'நிசப்தத்தில்', 'తర్వాతయినా', 'امیریکی']

# languages = ['bengali']

# for lang in languages:

folder = 'gt_CHIPS'


   
img_dir = DATA_DIR + 'test/images/'

output_dir = f'{RESULTS_DIR}intermediate/{folder}/'

if(folder=='gt_CHIPS1'):
    txt_dir = f'/data/BADRI/OCR/data/CHIPS/test/txt/'
else:
    txt_dir = f'/data/BADRI/OCR/results/detection/txt/{folder}/'



data = {}


hello = False
for lang in languages:
    for file in sorted(os.listdir(img_dir), key=natural_sort_key):
        
        if(file.split('_')[0]!=lang) and hello==False:
            continue
        elif(file.split('_')[0]!=lang and hello==True):
            with open(output_dir + lang + '/labels.json', 'w') as outfile:
                json.dump(data, outfile, indent=4)
            hello=False
            break            
        else: 
            if(hello==False):
                if not os.path.exists(output_dir + lang + '/images/'):
                    os.makedirs(output_dir + lang + '/images/')
                data = {}
                hello = True
            if(folder=='gt_CHIPS1'):
                df = pd.read_csv(txt_dir + file[:-4] + '.txt', sep=' ', names=['label', 'x1', 'y1', 'x2', 'y2'])
                df['id'] = df.index + 1  
            else:
                df = pd.read_csv(txt_dir + file[:-4] + '.txt', sep=' ', names=['id', 'x1', 'y1', 'x2', 'y2'])
            img = cv2.imread(img_dir + file)
            
            height, width, _ = img.shape
            
            for _, row in df.iterrows():
                
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                cropped_image = img[y1:y2, x1:x2]
                name = file[:-4] +  '_' + str(int(row['id'])) + '.jpg'
                cv2.imwrite(output_dir + lang +  '/images/' +  name , cropped_image)
                if(folder=='gt_CHIPS1'):
                    data[name] = row['label']
                else:
                    data[name] = placeholders[lang_range[lang]]
                
with open(output_dir + 'urdu' + '/labels.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

            
        