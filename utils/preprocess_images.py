import cv2
import argparse
import os
import shutil



def preprocess(file_path,border_x,border_y):
    img=cv2.imread(file_path)
    y,x, _=img.shape
    border_cut_y=int(border_y/100*y)
    border_cut_x=int(border_x/100*x)
    img=img[border_cut_y:y-border_cut_y,border_cut_x:x-border_cut_x]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    iy,iw=gray.shape
    thresh= cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    blur=cv2.GaussianBlur(gray,(13,13),100)
    thresh_inv=cv2.threshold(blur,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    cnts=cv2.findContours(thresh_inv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if len(cnts)==2 else cnts[1]
    cnts=sorted(cnts,key=lambda x:cv2.boundingRect(x)[1])
    xl,yl,xh,yh=0,0,0,0
    for c in cnts:
        x,y,w,h=cv2.boundingRect(c)
        if not (((abs(x-0)<5 or abs(x-iw)<5) or (abs(y-0)<5 or abs(y-iy)<5)) and h<30 and w<50):
            if xh==0:
                xl,yl,xh,yh=x,y,x+w,y+h
            else:
                xl=min(xl,x)
                yl=min(yl,y)
                xh=max(xh,x+w)
                yh=max(yh,y+h)
                
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
    crp=thresh[yl:yh,xl:xh]
    return crp

def main(args):
    op_folder = args.output_folder_name
    
    if not os.path.exists(op_folder):
        os.makedirs(op_folder)
    
    img_files=os.listdir(args.image_folder)
    for img in img_files:
        img_name=img.split('.')[0]
        output_image=preprocess(os.path.join(args.image_folder,img),args.border_cut_x,args.border_cut_y)
        # print(output_image)
        try:
            cv2.imwrite(op_folder+f'/{img_name}.jpg',output_image)
        except:
            shutil.copy(os.path.join(args.image_folder,img),op_folder+f'/{img_name}.jpg')
            print(img_name)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--image_folder", type=str, default=None, help="path to the input img file")
    parser.add_argument("-o", "--output_folder_name", type=str, default="OUTPUT", help="path to the output img directory")
    parser.add_argument("-c", "--border_cut_x", type=int, default=3.5, help="percent of border cut of word level image at x-axis")
    parser.add_argument("-r", "--border_cut_y", type=int, default=3.5, help="percent of border cut of word level image at y-axis")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
