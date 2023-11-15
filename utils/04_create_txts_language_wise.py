import os
import shutil

languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']

input_dir = '/data/BADRI/OCR/results/detection/txt/'

for lang in languages:
    if not os.path.exists(f'{input_dir}languages/{lang}/'):
        os.makedirs(f'{input_dir}languages/{lang}/')

for file in os.listdir(input_dir + 'finetuned_CHIPS1/'):
    lang = file.split('_')[0]
    shutil.copy(input_dir + 'finetuned_CHIPS1/' + file, f'{input_dir}languages/{lang}/'+ file)