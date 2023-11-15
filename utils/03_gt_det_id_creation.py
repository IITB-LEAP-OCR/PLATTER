import os
import pandas as pd


txt_dir = '/data/BADRI/OCR/data/CHIPS1/test/txt/'
out_dir = '/data/BADRI/OCR/results/detection/txt/gt_CHIPS1/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file in os.listdir(txt_dir):
    df = pd.read_csv(txt_dir + file[:-4] + '.txt', sep=' ', names=['label', 'x1', 'y1', 'x2', 'y2'])
    df['id'] = df.index + 1
    df = df[['id', 'x1', 'y1', 'x2', 'y2' ]]
    df.to_csv(out_dir + file[:-4] + '.txt', sep=' ', index=False, header=False)