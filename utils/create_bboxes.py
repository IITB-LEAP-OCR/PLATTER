import cv2
import os
import pandas as pd


def draw_bboxes(txt_path, image_path):
    df = pd.read_csv(txt_path, sep=' ', names=['label', 'x1', 'y1', 'x2', 'y2'])
    img = cv2.imread(image_path)
    
    for _, row in df.iterrows():
        cv2.rectangle(img, (row['x1'], row['y1']), (row['x2'], row['y2']), (255, 0, 0), 2)
        # exit()
        
    print("hello")
    cv2.imwrite('output3.jpg', img)
    
    
    
def main():
    image_path = '/data/BADRI/RECOGNITION/PAGE_LEVEL/CHIPS/test/images/bengali_page_1.jpg'
    txt_path = '/data/BADRI/OCR/results/pretrained_crnn_vgg16_bn/bengali/bengali_page_1.txt'
    
    draw_bboxes(txt_path, image_path)
    
    
main()