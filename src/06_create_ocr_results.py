import os
import re
import pandas as pd
import cv2
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']
models = ['parseq', 'crnn_vgg16_bn', 'master', 'vitstr_small', 'crnn_mobilenet_v3_small', 'sar_resnet31']
folders = ['gt_CHIPS1', 'finetuned_CHIPS1', 'pretrained_CHIPS1']

rec_results_dir = '/data/BADRI/OCR/results/recognition/'
det_results_dir = '/data/BADRI/OCR/results/detection/'

img_path = '/data/BADRI/OCR/data/CHIPS_1/test/images/'

for folder in folders:
    for model in models:

        out_txt_path = f'/data/BADRI/OCR/results/ocr/{folder}/{model}/'
        det_txt_path = f'/data/BADRI/OCR/results/detection/txt/{folder}/'
        rec_txt_path = f'/data/BADRI/OCR/results/recognition/{folder}/{model}/{lang}.txt'
      
        if not os.path.exists(out_txt_path):
            os.makedirs(out_txt_path)
        
        value = 0
        lang = languages[value]
        
        for file in tqdm(sorted(os.listdir(det_txt_path), key=natural_sort_key)):
            img = cv2.imread(img_path + file[:-4] + '.jpg')
            height, width, _ = img.shape
            name = file[:-4] + '_'
            
            df = pd.read_csv(rec_txt_path, sep=' ', names=['file', 'pred'])
            subset_df = df[df['file'].str.startswith(name)].reset_index(drop=True)
            det_df = pd.read_csv(det_txt_path + file, sep=' ', names=['label', 'x1', 'y1', 'x2', 'y2'])
            det_df['id'] = det_df.index + 1
            result_df = pd.concat([det_df, subset_df['pred']], axis=1)
            
            result_df['x1'] = result_df['x1']*width
            result_df['x2'] = result_df['x2']*width
            result_df['y1'] = result_df['y1']*height
            result_df['y2'] = result_df['y2']*height
            
            result_df = result_df[['pred', 'x1', 'y1', 'x2', 'y2']]
            
            result_df['x1'] = result_df['x1'].astype(int)
            result_df['x2'] = result_df['x2'].astype(int)
            result_df['y1'] = result_df['y1'].astype(int)
            result_df['y2'] = result_df['y2'].astype(int)
            
            result_df.to_csv(out_txt_path + file, sep=' ', index=False, header=False)
            
            if(height!=1024):
                value +=1
                lang = languages[value]
                txt_path = f'/data/BADRI/OCR/results/recognition/{folder}/{model}/{lang}.txt'

