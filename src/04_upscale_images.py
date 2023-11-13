import cv2
# import argparse
import os
from tqdm import tqdm

def main(output_folder_name, image_folder, image_h):    
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    
    img_files=os.listdir(image_folder)
    for img_file in tqdm(img_files):
        img_name=img_file.split('.')[0]
        img=cv2.imread(os.path.join(image_folder,img_file))
        y,x=img.shape[:2]
        new_x=int(image_h/y*x)
        output_image=cv2.resize(img,(new_x,image_h))
        cv2.imwrite(output_folder_name+f'/{img_name}.jpg',output_image)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument("-i", "--image_folder", type=str, default=None, help="path to the input img file")
#     parser.add_argument("-o", "--output_folder_name", type=str, default="OUTPUT", help="path to the output img directory")
#     parser.add_argument("-l", "--image_h", type=int, default=295, help="height of the image")
#     args = parser.parse_args()
#     return args


languages = ['bengali', 'gujarati', 'hindi', 'kannada', 'malayalam', 'gurumukhi', 'odia', 'urdu', 'tamil', 'telugu']

folders = ['finetuned_CHIPS1', 'gt_CHIPS1', 'pretrained_CHIPS1']

for folder in folders:
    for lang in languages:
        INPUT_DIR = f'./../../intermediate/{folder}/{lang}/images/'
        OUTPUT_DIR = f'./../../intermediate_2/{folder}/{lang}/images/'   
        main(OUTPUT_DIR, INPUT_DIR, 295)
        os.system(f'cp ./../../intermediate/{folder}/{lang}/labels.json ./../../intermediate_2/{folder}/{lang}/labels.json')

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
