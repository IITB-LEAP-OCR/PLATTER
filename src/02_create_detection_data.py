import os
import json
import cv2
import re
import pandas as pd
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']


DATA_DIR = '/data/BADRI/OCR/PLATTER_OF_CHIPS/data/CHIPS/'
sets = ['test', 'val', 'train']

    
for typeset in sets:
    
    input_dir = os.path.join(DATA_DIR, typeset)

    final_data = {}
    lang_data = {}
    file_counts = 0
    index = 0
    language = languages[index]
    for file in tqdm(sorted(os.listdir(input_dir +'/txt/'), key=natural_sort_key)):
        lang_value = file.split('_')[0]
        if(lang_value != language):
            with open(os.path.join(input_dir,  language + '.json'), "w") as json_file:
                json.dump(lang_data, json_file, indent=4)
            lang_data = {}
            print(language, file_counts)
            file_counts = 0
            index += 1
            language = languages[index]
        file_counts +=1
        df = pd.read_csv(os.path.join(input_dir, 'txt', file), sep=' ', names=['label', 'x1', 'y1', 'x2', 'y2'])
        img = cv2.imread(os.path.join(input_dir, 'images', file[:-4] + '.jpg'))
        
        image_data = {}
        height, width, _ = img.shape
        image_data['img_dimensions'] = (height, width)
        image_data['img_hash'] = file[:-4] + '.jpg'
        
        bbox_data = []
        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            bbox_data.append([[x1, y1],[x2, y1],[x2, y2],[x1, y2]])
            
        image_data['polygons'] = bbox_data
        final_data[file[:-4] + '.jpg'] = image_data
        lang_data[file[:-4] + '.jpg'] = image_data
    
    
    with open(os.path.join(input_dir,  language + '.json'), "w") as json_file:
        json.dump(lang_data, json_file, indent=4)
    
    #Save data in json format
    with open(os.path.join(input_dir, 'labels.json'), "w") as json_file:
        json.dump(final_data, json_file, indent=4)
        