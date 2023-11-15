import os
import re
import pandas as pd
import cv2
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']
models = ['parseq', 'crnn_vgg16_bn', 'master', 'vitstr_small', 'crnn_mobilenet_v3_small', 'sar_resnet31']
folders = ['finetuned_CHIPS1', 'pretrained_CHIPS1']

rec_results_dir = '/data/BADRI/OCR/results/recognition/'
det_results_dir = '/data/BADRI/OCR/results/detection/'
img_path = '/data/BADRI/OCR/data/CHIPS1/test/images/'

# languages = ['bengali']
# models = ['parseq']
folders = ['pretrained_CHIPS1']

for folder in folders:
    for model in models:
        

        out_txt_path = f'/data/BADRI/OCR/results/ocr/{folder}/{model}/'
        det_txt_path = f'/data/BADRI/OCR/results/detection/txt/{folder}/'
        
      
        if not os.path.exists(out_txt_path):
            os.makedirs(out_txt_path)
        
        
        for file in tqdm(sorted(os.listdir(det_txt_path), key=natural_sort_key)):
            lang = file.split('_')[0]
            rec_txt_path = f'/data/BADRI/OCR/results/recognition/{folder}/{model}/{lang}.txt'
            name = file[:-4] + '_'
            
            df = pd.read_csv(rec_txt_path, sep=' ', names=['file', 'pred'])
            subset_df = df[df['file'].str.startswith(name)].reset_index(drop=True)
            det_df = pd.read_csv(det_txt_path + file, sep=' ', names=['id', 'x1', 'y1', 'x2', 'y2'])
            result_df = pd.concat([det_df, subset_df['pred']], axis=1)
                        
            result_df = result_df[['pred', 'x1', 'y1', 'x2', 'y2']]
            
            result_df.to_csv(out_txt_path + file, sep=' ', index=False, header=False)

