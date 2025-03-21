import cv2
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm
import json


def read_image(img_path, label_len, img_h=64, char_w=32):
    valid_img = True
    
    try:
        img = cv2.imread(img_path, 0) # reading the image in grayscale
        y,x=img.shape
        img=img[5:y-10,10:x-10]  # cropping the boundaries to remove noises at the boundaries
        iy,iw=img.shape
        thresh= cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # binared image 
        blur=cv2.GaussianBlur(img,(13,13),100) # blur the image to remove the noises
        thresh_inv=cv2.threshold(blur,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] # inverted image
        cnts=cv2.findContours(thresh_inv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #finding contours in the image
        cnts=cnts[0] if len(cnts)==2 else cnts[1]
        cnts=sorted(cnts,key=lambda x:cv2.boundingRect(x)[1])
        xl,yl,xh,yh=0,0,0,0
        for c in cnts:
            x,y,w,h=cv2.boundingRect(c)
            if xh==0:
                xl,yl,xh,yh=x,y,x+w,y+h
            else:
                xl=min(xl,x)  # mapping all the contours to get the main image
                yl=min(yl,y)
                xh=max(xh,x+w)
                yh=max(yh,y+h)

        img=img[yl:yh+5,xl:xh+5]
        curr_h, curr_w = img.shape
        modified_w = int(curr_w * (img_h / curr_h))
        img=cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=5)

        # Remove outliers
        if ((modified_w / label_len) < (char_w / 3)) | ((modified_w / label_len) > (3 * char_w)):
            valid_img = False
        else:
            # Resize image so height = img_h and width = char_w * label_len
            img_w = label_len * char_w
            img = cv2.resize(img, (img_w, img_h))
            img=cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    except AttributeError:
        valid_img = False
        print('Error at',img_path)

    return img, valid_img

def main(images_folder, label_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    
    with open(label_path, 'r') as json_file:
        data = json.load(json_file)
    
    df = pd.DataFrame.from_dict(data, orient ='index').reset_index()
    df.columns = ['path', 'label']
    
    for _, row in tqdm(df.iterrows()):

        img,_=read_image(images_folder + row['path'],len(row['label']),32,16)

        cv2.imwrite(output_path+row['path'],img)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--img_path", type=str, default=None, help="path to the input img file")
    parser.add_argument("-l", "--label", type=str, default=None, help="label fo the img")
    parser.add_argument("-o", "--output_folder_name", type=str, default="OUTPUT", help="path to the output img ")
    # parser.add_argument("--char_h", type=int, default=32, help="height of the image")
    # parser.add_argument("--char_w", type=int, default=16, help="width of the image")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = parse_args()
    
    LANGUAGE = 'bengali'
    
    root_dir = '/data/BADRI/OCR/intermediate/gt_chips_1/'
    
    
    
    sets = ['test', 'val', 'train']
    sets = ['images']
    for value in sets:
        images_folder = root_dir + LANGUAGE + '/' + value + '/'
        label_path = root_dir + LANGUAGE + '/'  + 'labels.json'
        output_path = root_dir + LANGUAGE + '_new/' + value + '/'
        main(images_folder, label_path, output_path)