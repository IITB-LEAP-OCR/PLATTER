import os
import re
import pandas as pd
import cv2
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']
languages = ['bengali']
models = ['parseq', 'crnn_vgg16_bn', 'master', 'vitstr_small', 'crnn_mobilenet_v3_small', 'vitstr_base', 'sar_resnet31']

models = ['parseq']


folder_name = 'finetuned_CHIPS_1'

rec_results_dir = '/data/BADRI/OCR/results/recognition/'

det_results_dir = '/data/BADRI/OCR/results/detection/'



for model in models:
    # for lang in languages:
    out_txt_path = f'/data/BADRI/OCR/results/ocr/{folder_name}/{model}/'
    
    det_txt_path = f'/data/BADRI/OCR/results/detection/txt/{folder_name}/'
    img_path = f'/data/BADRI/OCR/results/detection/images/{folder_name}/'

    
    if not os.path.exists(out_txt_path):
        os.makedirs(out_txt_path)
    
    # df = pd.read_csv(txt_path, sep=' ', names=['file', 'pred'])
    value = 0
    lang = languages[value]
    txt_path = f'/data/BADRI/OCR/results/recognition/{folder_name}/{model}/{lang}_lang.txt'
    for file in tqdm(sorted(os.listdir(det_txt_path), key=natural_sort_key)):
        img = cv2.imread(img_path + file[:-4] + '.jpg')
        height, width, _ = img.shape
        

        name = file[:-4] + '_'
        df = pd.read_csv(txt_path, sep=' ', names=['file', 'pred'])
        subset_df = df[df['file'].str.startswith(name)].reset_index(drop=True)
        det_df = pd.read_csv(det_txt_path + file, sep=' ', names=['id', 'x1', 'y1', 'x2', 'y2'])
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
            txt_path = f'/data/BADRI/OCR/results/recognition/{folder_name}/{model}/{lang}_synth.txt'
        # print("done")
    # print("done")
