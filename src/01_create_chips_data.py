import cv2
import os
import numpy as np
import random
import math
import json
import pandas as pd

from config import *

def preprocess(file_path,border_x,border_y):
    
    img=cv2.imread(file_path)
    y,x_p,_=img.shape
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(13,13),100)
    thresh=cv2.threshold(blur,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    black_pixel_counts = np.sum(thresh == 0, axis=0)

    print(len(black_pixel_counts))
    # print(black_pixel_counts)

    median = np.median(black_pixel_counts)
    print('median :',median)
    data={}
    for i in range(len(black_pixel_counts)):
        if black_pixel_counts[i]<=median+5:
            black_pixel_counts[i]=0
        data[str(i)]=black_pixel_counts[i]

    l_flag=1
    r_flag=1
    for i in range(len(black_pixel_counts)//2):
        if i<=10:
            continue
        if l_flag and black_pixel_counts[i]==0:
            l_flag=0
        if not l_flag and black_pixel_counts[i]!=0:
            break
    if i == len(black_pixel_counts)//2:
        l=0
    else:
        l=i

    for i in range(len(black_pixel_counts)-1,len(black_pixel_counts)//2,-1):
        if r_flag and black_pixel_counts[i]==0:
            r_flag=0
        if not r_flag and black_pixel_counts[i]!=0:
            break
    if i ==len(black_pixel_counts)//2:
        r=len(black_pixel_counts)
    else:
        r=i

    print(l,r)

    y,x,_=img.shape
    img=img[0:y,l:r]
    # cv2.imshow('cropped image',img)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_inv=cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,2))
    # dilate=cv2.dilate(thresh_inv, kernel, iterations=20)
    dilate = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel,iterations=10)
    # cv2.imshow('dilated image',dilate)
    cnts=cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    m=[]
    print('length of cnts :',len(cnts[0]))
    for i in cnts[0]:
        if len(m)<len(i):
            m=i
    x,y,w,h=cv2.boundingRect(m)
    pad=10
    if x-pad>=0:
        x=x-pad
        print('x :',x)

    final_image = img[y:y+h,x:-1]
    # cv2.imwrite(os.path.basename(file_path).split('.')[0]+'_cropped.jpg',final_image)
    # cv2.imshow('final Image', final_image)
    # # cv2.imshow('morphed Image', morph)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return final_image

def save_image(final_image,img_bbox,file_name,output_path):
    i=np.concatenate(final_image)
    i = i.astype("uint8")
    cv2.imwrite(os.path.join(output_path,'images',file_name)+'.jpg',i)
    with open(os.path.join(output_path,'txt',file_name)+'.txt','w+',encoding='utf8') as f:
        f.writelines([i+'\n' for i in img_bbox])

def gen_images(language, input_folder,output_folder,image_map, saved_pages):
    
    final_image=[]
    img_bbox=[]
    sentence_img=[]
    n_lines=0
    line_x=int(MAX_WORD_H/3)
    saved_pages=0
    max_lines=math.floor((PAGE_H-UPPER_PADDING)/(MAX_WORD_H+SPACE_Y))
    page_left_from_bottom=PAGE_H-UPPER_PADDING-((MAX_WORD_H+SPACE_Y)*max_lines)
    sentence_img.append(np.ones((MAX_WORD_H, int(MAX_WORD_H/3)))*255)
    skipped_words=[]
    
    for index,img_path in enumerate(os.listdir(input_folder)):
        try:
            img=preprocess(os.path.join(input_folder,img_path),BORDER_CUT_X,BORDER_CUT_Y)
            y,x=img.shape[:2]
            w_h=random.randint(MIN_WORD_H,MAX_WORD_H)
            data_stats[w_h] +=1
            new_x=int(w_h/y*x)
            if (int(MAX_WORD_H/3)+new_x+SPACE_X)>PAGE_W:
                w_h=MIN_WORD_H
                new_x=int(w_h/y*x)
            img=cv2.resize(img,(new_x,w_h))
            y,x=img.shape[:2]
            img=np.concatenate([img,np.ones((MAX_WORD_H-w_h,x))*255])
            line_x+=x+SPACE_X
                
            y,x=img.shape[:2]

            if line_x>PAGE_W:
                sentence_img = np.hstack(sentence_img)
                yt,xt=sentence_img.shape[:2]

                if xt>PAGE_W:
                    sentence_img=[]
                    sentence_img.append(np.ones((MAX_WORD_H, int(MAX_WORD_H/3)))*255)
                    line_x=int(MAX_WORD_H/3)
                    print(image_map[img_path],"skipped due to large width")
                    skipped_words.append(image_map[img_path])
                    continue
                    
                residual=PAGE_W-sentence_img.shape[1]
                sentence_img=np.hstack([sentence_img,np.ones((MAX_WORD_H,residual))*255])
                sentence_img=cv2.resize(sentence_img,(PAGE_W,MAX_WORD_H))
                sentence_img=np.concatenate([sentence_img,np.ones((SPACE_Y,PAGE_W))*255])
                final_image.append(sentence_img)
                n_lines+=1
                if n_lines%max_lines==0:
                    saved_pages+=1
                    final_image.append(np.ones((page_left_from_bottom,PAGE_W))*255)
                    final_image.insert(0,np.ones((UPPER_PADDING,PAGE_W))*255)
                    save_image(final_image,img_bbox,f'{language}_page_{saved_pages}',output_folder)
                    final_image=[]
                    img_bbox=[]
                sentence_img=[]
                sentence_img.append(np.ones((MAX_WORD_H, int(MAX_WORD_H/3)))*255)
                line_x=int(MAX_WORD_H/3)+new_x+SPACE_X
                # print('NEW LINE')
            bbox_x1=max(0,line_x-new_x-SPACE_X - 8)
            bbox_y1=max(0, (MAX_WORD_H+SPACE_Y)*(n_lines%max_lines) - 8 + UPPER_PADDING)
            bbox_x2=line_x-SPACE_X + 8
            bbox_y2=(MAX_WORD_H+SPACE_Y)*((n_lines%max_lines)+1)-SPACE_Y - (MAX_WORD_H-w_h) + 8 + UPPER_PADDING
            t_bbox=f'{image_map[img_path]} {bbox_x1} {bbox_y1} {bbox_x2} {bbox_y2}'
        except Exception as A:
            print(A)
            continue
        sentence_img.append(img)
        sentence_img.append(np.ones((MAX_WORD_H, SPACE_X))*255)
        img_bbox.append(t_bbox)

    if len(sentence_img)>0:
        try:
            sentence_img = np.hstack(sentence_img)
            residual=PAGE_W-sentence_img.shape[1]
            sentence_img=np.hstack([sentence_img,np.ones((MAX_WORD_H,residual))*255])
            sentence_img=cv2.resize(sentence_img,(PAGE_W,MAX_WORD_H))
            final_image.append(sentence_img)
            n_lines+=1
            
            saved_pages+=1
            final_image.append(np.ones((page_left_from_bottom,PAGE_W))*255)
            save_image(final_image,img_bbox,f'{language}_page_{saved_pages}',output_folder)
        except:
            pass


    with open(os.path.join(output_folder,'skipped_words')+'.txt','w+',encoding='utf8') as f:
        f.writelines([i+'\n' for i in skipped_words])
    print('SKIPPED WORDS LIST SAVED')
    print('PROCESS FINISHED')



data_stats = {}

for value in range(31,65):
    data_stats[value]=0

if __name__ == "__main__":
    

    
    for language in LANGUAGES:
        input_folder = INPUT_FOLDER + language + '/'

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        
        for typeset in SETS:
            image_map_file = input_folder + typeset + '/' + typeset + '_gt.txt'
            data = pd.read_csv(image_map_file, sep='\t', header=None, names=['Image', 'Category'], encoding='utf-8')
            data['Image'] = data['Image'].apply(lambda x: x.split('/')[-1])
            image_map = data.set_index('Image').to_dict()['Category']
            
            try:
                saved_pages = len(os.listdir(OUTPUT_FOLDER + typeset + '/images/'))
            except:
                if not os.path.exists(OUTPUT_FOLDER + typeset + '/images/'):
                    os.makedirs(OUTPUT_FOLDER + typeset + '/images/')
                    os.makedirs(OUTPUT_FOLDER + typeset + '/txt/')
                saved_pages = 0
            img=gen_images(language, input_folder + typeset + '/images/', OUTPUT_FOLDER + typeset + '/', image_map, saved_pages)
            
            
    
    with open('data_stats.json', 'w', encoding='utf-8') as f:
        json.dump(data_stats, f, ensure_ascii=False, indent=4)
 
 
 
 
# def parse_args():
#     parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument("-i", "--input_folder", type=str, default=None, help="path to the input folder of the image file")
#     parser.add_argument("-o", "--output_folder", type=str, default="", help="path to the output img directory")
#     parser.add_argument("-m", "--image_map", type=str, default=None, help="Path to the json file consisting of image map")
#     parser.add_argument("-l", "--page_h", type=int, default=1024, help="height of whole page")
#     parser.add_argument("-w", "--page_w", type=int, default=1024, help="width of whole page")
#     parser.add_argument("-q", "--min_word_h", type=int, default=32, help="min_height of each word")
#     parser.add_argument("-p", "--max_word_h", type=int, default=64, help="max_height of each word")
#     parser.add_argument("-x", "--space_x", type=int, default=64, help="space between each word")
#     parser.add_argument("-y", "--space_y", type=int, default=32, help="space between each line")
#     parser.add_argument("-c", "--border_cut_x", type=int, default=3.5, help="percent of border cut of word level image at x-axis")
#     parser.add_argument("-r", "--border_cut_y", type=int, default=3.5, help="percent of border cut of word level image at y-axis")
#     args = parser.parse_args()
#     return args
 
 
#     # args = parse_args()
#     # main(args)