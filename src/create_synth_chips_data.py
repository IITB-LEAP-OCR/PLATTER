# import required packages
from config import Config
import pickle as pkl
from generate_images import ImgGenerator
import numpy as np
import cv2
import os
import argparse
import math
import random
from tqdm import tqdm

from config import *

def save_image(final_image,img_bbox,file_name,output_path):
    i=np.concatenate(final_image)
    i = i.astype("uint8")
    # print(f'SAVING {file_name}')
    cv2.imwrite(os.path.join(output_path,'images',file_name)+'.jpg',i)
    with open(os.path.join(output_path,'txt',file_name)+'.txt','w+',encoding='utf8') as f:
        f.writelines([i+'\n' for i in img_bbox])

def gen_images(generator, input_words, language,output_folder, saved_pages):

    final_image, img_bbox, sentence_img, skipped_words = [], [], [], []
    n_lines=0
    line_x=int(MAX_WORD_H/3)
    max_lines=math.floor(PAGE_H/(MAX_WORD_H+SPACE_Y))
    page_left_from_bottom=PAGE_H-((MAX_WORD_H+SPACE_Y)*max_lines)
    sentence_img.append(np.ones((MAX_WORD_H, int(MAX_WORD_H/3)))*255)
    
    for word in tqdm(input_words):
        try:
            generated_imgs, _, word_labels = generator.generate(word_list=[word])
            img=generated_imgs[0]
            img = img[:, img.sum(0) < 31.5]*255
            y,x=img.shape[:2]
            w_h=random.randint(MIN_WORD_H,MAX_WORD_H-1)
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
                    print(word,"skipped due to large width")
                    skipped_words.append(word)
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
                    save_image(final_image,img_bbox,f'{language}_page_{saved_pages}',output_folder)
                    final_image=[]
                    img_bbox=[]
                sentence_img=[]
                sentence_img.append(np.ones((MAX_WORD_H, int(MAX_WORD_H/3)))*255)
                line_x=int(MAX_WORD_H/3)+new_x+SPACE_X
                # print('NEW LINE')
            bbox_x1=max(0, line_x-new_x-SPACE_X - 8)
            bbox_y1=max(0, (MAX_WORD_H+SPACE_Y)*(n_lines%max_lines) - 8)
            bbox_x2=line_x-SPACE_X + 8
            bbox_y2=(MAX_WORD_H+SPACE_Y)*((n_lines%max_lines)+1)-SPACE_Y - (MAX_WORD_H-w_h) + 8
            
            t_bbox=f'{word} {bbox_x1} {bbox_y1} {bbox_x2} {bbox_y2}'
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
    
    
def main(args):
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder + 'images/')
        os.makedirs(args.output_folder + 'txt/')
        
    # loading the model
    config = Config
    print('LOADING THE MODEL')
    with open(config.data_file, 'rb') as f:  # data file path
        char_map = pkl.load(f)
    char_map=char_map['char_map']
    generator = ImgGenerator(checkpt_path=args.model_path, config=config, char_map=char_map)
    
    with open(args.input_file,'r') as f:
        input_words=f.read().split('\n')

    language = 'telugu'
    gen_images(generator, input_words, language,args.output_folder, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_file", type=str, default=None, help="path to the input img file")
    parser.add_argument("-o", "--output_folder", type=str, default="", help="path to the output img directory")
    parser.add_argument("-m", "--model_path", type=str, default='./weights/model_checkpoint_epoch_100.pth.tar', help="path to the model")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)