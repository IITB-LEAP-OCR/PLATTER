import pandas as pd
import os


def process_word(x):
    try:
        for i in range(len(x)):
            try:
                if(x[i]==x[i+1] and i>3):
                    if(x[i+1]==x[i+2]):
                        if(x[i+2]==x[i+3]):
                            if(x[i+3]==x[i+4]):
                                return x[:i]
            except:
                return x[:i+2]
    except:
        return x
    

input_dir = '/data/BADRI/OCR/results/recognition/finetuned_CHIPS1/master/'
output_dir = '/data/BADRI/OCR/results/recognition/finetuned_CHIPS1/master_cleaned/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for file in os.listdir(input_dir):
    df = pd.read_csv(input_dir + file, sep=' ', names=['id', 'label'])
    print(file)
    df['label'] = df['label'].apply(lambda x: process_word(x))
    df.to_csv(output_dir + file, sep=' ', index=False, header=False)