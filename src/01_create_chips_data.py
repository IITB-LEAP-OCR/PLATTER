import cv2
import os
import numpy as np
import random
import math
import json
import pandas as pd

from config import *

def preprocess_1(file_path,border_x,border_y):
    """
    Preprocesses and Crops the Input Word level Image.

    Arguments:
    file_path: Path to the image file.
    border_x: Percentage of the border to be cut along x-axis
    border_y: Percentage of the border to be cut along y-axis

    Returns:
    numpy.ndarray: Preprocessed image if successful, None otherwise.
    """
    # Load the Image and apply Border Cutting on it
    img=cv2.imread(file_path)
    y,x=img.shape[:2]
    border_cut_y=int(border_y/100*y)
    border_cut_x=int(border_x/100*x)
    img=img[border_cut_y:y-border_cut_y,border_cut_x:x-border_cut_x]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    iy,iw=gray.shape
    # Binarize the Image and Get Contours from the Image
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
                
    crp=thresh[yl:yh,xl:xh]
    return crp

def preprocess_2(file_path):
    """
    Preprocesses and Crops the Input Word level Image.

    Arguments:
    file_path: Path to the image file.

    Returns:
    numpy.ndarray: Preprocessed image if successful, None otherwise.
    """
    # Read image
    img=cv2.imread(file_path)

    # Crop borders
    y,x_p,_=img.shape
    img=img[0:y,BORDER_CUT_X:x_p]
    x_p-=BORDER_CUT_X

    # Convert to grayscale and apply Gaussian blur
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(13,13),100)

    # Apply Otsu's thresholding
    thresh=cv2.threshold(blur,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # Calculate black pixel counts
    black_pixel_counts = np.sum(thresh == 0, axis=0)
    median = np.median(black_pixel_counts)
    data={}
    # Remove noise on both sides
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

    for i in range(len(black_pixel_counts)-1,len(black_pixel_counts)//2,-9):
        if r_flag and black_pixel_counts[i]==0:
            r_flag=0
        if not r_flag and black_pixel_counts[i]!=0:
            break
    if i ==len(black_pixel_counts)//2:
        r=len(black_pixel_counts)
    else:
        r=i

    # Crop image based on computed boundaries
    y_m,x_m=thresh.shape[:2]
    border_cut_y=int(BORDER_CUT_Y/100*y_m)
    img=thresh[border_cut_y:y_m-border_cut_y,l:r]
    y_m-=border_cut_y

    # Invert threshold and find contours
    thresh_inv=cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    cnts=cv2.findContours(thresh_inv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if len(cnts)==2 else cnts[1]
    cnts=sorted(cnts,key=lambda x:cv2.boundingRect(x)[1])

    # Find bounding box of the text
    xl,yl,xh,yh=0,0,0,0
    for c in cnts:
        x,y,w,h=cv2.boundingRect(c)
        if not (((abs(x-0)<5 or abs(x-x_m)<5) or (abs(y-0)<5 or abs(y-y_m)<5)) and h<30 and w<50):
            if xh==0:
                xl,yl,xh,yh=x,y,x+w,y+h
            else:
                xl=min(xl,x)
                yl=min(yl,y)
                xh=max(xh,x+w)
                yh=max(yh,y+h)

    # Crop the final image
    final_image = img[yl:yh,xl:xh]
    
    return final_image

def save_image(final_image, img_bbox, file_name, output_path):
    """
    This function saves the final page level image along with corresponding text and bounding box information.
    
    Arguments:
    final_image: The final page level image.
    img_bbox: The corresponding text and the bounding box information corresponding to the final image.
    file_name: Name of the file to be saved.
    output_path: Path to the output directory.
    """
    # Color map for text
    text_colors = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    
    # Choose a random text color
    txt_color_index = random.randint(0, 3)
    
    # Concatenate final image
    concatenated_image = np.concatenate(final_image)
    concatenated_image = concatenated_image.astype("uint8")
    
    # Create a black mask
    black_mask = concatenated_image <= 127
    
    # Create result image
    result_image = np.ones((concatenated_image.shape[0], concatenated_image.shape[1], 3), dtype=np.uint8) * 255
    result_image[black_mask] = text_colors[txt_color_index]
    
    # Save image
    image_file_path = os.path.join(output_path, 'images', file_name) + '.jpg'
    
    cv2.imwrite(image_file_path, result_image)
    
    print("Image saved:", file_name)
    
    # Save bounding box information to text file
    with open(os.path.join(output_path, 'txt', file_name) + '.txt', 'w+', encoding='utf8') as txt_file:
        txt_file.writelines([line + '\n' for line in img_bbox])

def gen_images(language, input_folder,output_folder,image_map, saved_pages, sorted_image_map):
    """
    Generates images from input images and saves them in the output folder.

    Args:
        language (str): Language of the images.
        input_folder (str): Folder containing input images.
        output_folder (str): Folder where generated images will be saved.
        image_map (dict): Dictionary mapping image paths to their corresponding text.
        saved_pages (int): Counter for saved pages.
        sorted_image_map (dict): Sorted dictionary mapping image paths to their corresponding text.

    Returns:
        None
    """
    # Initialize lists and counters for image generation
    final_image=[]
    img_bbox=[]
    sentence_img=[]
    n_lines=0
    line_x=int(MAX_WORD_H/3)
    saved_pages=0
    max_lines=math.floor((PAGE_H-UPPER_PADDING)/(MAX_WORD_H+SPACE_Y))
    page_left_from_bottom=PAGE_H-UPPER_PADDING-((MAX_WORD_H+SPACE_Y)*max_lines)
    sentence_img.append(np.ones((MAX_WORD_H, int(MAX_WORD_H/3)))*255)
    sentence_text=' '
    skipped_words=[]
    used_files=[]

    # Generate random word height and calculate gap between words
    w_h=random.randint(MIN_WORD_H,MAX_WORD_H)
    font_sizes = list(range(MIN_WORD_H, MAX_WORD_H + 1))
    num_font_sizes = len(font_sizes)
    font_probabilities = [1 / num_font_sizes] * num_font_sizes
    weighted_avg_font_size = [size * prob for size, prob in zip(font_sizes, font_probabilities)]
    gap = int(w_h*weighted_avg_font_size[w_h-MIN_WORD_H]*2/3)
    if gap<MIN_SPACE_X:
        gap=MIN_SPACE_X
    if gap>w_h:
        gap=w_h
    
    # Loop through each image in the input folder
    for index,img_path in enumerate(os.listdir(input_folder)):
        try:
            # Skip if image is already used
            if img_path in used_files:
                continue

            # Preprocess the image
            if language.lower() in ['tamil','malayalam']:
                img_p=preprocess_2(os.path.join(input_folder,img_path))
            else:
                img_p=preprocess_1(os.path.join(input_folder,img_path), BORDER_CUT_X, BORDER_CUT_Y)
            y,x=img_p.shape[:2]
            data_stats[w_h] +=1
            new_x=int(w_h/y*x)
            img=cv2.resize(img_p,(new_x,w_h))
            y,x=img.shape[:2]

            # Concatenate with white space to make the height consistent
            img=np.concatenate([img,np.ones((MAX_WORD_H-w_h,x))*255])
            line_x+=x+gap

            # Keep track of used files
            used_files.append(img_path)

            # If the line exceeds page width, start a new line
            if line_x>PAGE_W:
                # Handle cases where smallest word fits in the remaining space
                # to avoid leaving a large gap at the end of the line

                smallest_word_index = 0
                skipped_images = 0
                line_x-=(x+gap)
                t_sentence_img=sentence_img

                while skipped_images<=5:
                    try:
                        sentence_img = np.hstack(t_sentence_img)
                        yt,xt=sentence_img.shape[:2]
                        
                        # Load the Smalles Image from the sorted image map
                        smallest_image, smallest_word = sorted_image_map[smallest_word_index]

                        # Skip the Image if already used
                        if smallest_image in used_files:
                            sorted_image_map.pop(smallest_word_index)
                            continue

                        # Preprocess the smallest image
                        if language.lower() in ['tamil','malayalam']:
                            t_img=preprocess_2(os.path.join(input_folder,smallest_image))
                        else:
                            t_img=preprocess_1(os.path.join(input_folder,smallest_image), BORDER_CUT_X, BORDER_CUT_Y)
                        
                        y,x=t_img.shape[:2]

                        # Skip if word height is smaller than the minimum word height
                        if y<w_h:
                            # print('Skipped')
                            sorted_image_map.pop(smallest_word_index)
                            used_files.append(smallest_image)
                            skipped_words.append(smallest_word)
                            skipped_images+=1
                            # cv2.imwrite('/data/circulars/DATA/TACTFUL/skipped_images/'+smallest_image,t_img)
                            continue

                        new_x=int(w_h/y*x)
                        remaining_sentence_width=PAGE_W-xt-8

                        # print('remaining_sentence_width :',remaining_sentence_width,'new_x :',new_x, end = ', ')

                        # If the word fits, add it to the sentence
                        if new_x+gap<remaining_sentence_width:
                            # print('Done')
                            sorted_image_map.pop(smallest_word_index)
                            used_files.append(smallest_image)
                            t_img=cv2.resize(t_img,(new_x,w_h))
                            y,x=t_img.shape[:2]

                            # Concatenate with white space to make the height consistent
                            t_img=np.concatenate([t_img,np.ones((MAX_WORD_H-w_h,x))*255])
                            line_x+=x+gap

                            # Append the word image and update the sentence text
                            t_sentence_img.append(t_img)
                            t_sentence_img.append(np.ones((MAX_WORD_H, gap))*255)
                            sentence_text+=smallest_word
                            sentence_img = np.hstack(t_sentence_img)
                            
                            # Calculate bounding box coordinates and append to the list
                            bbox_x1=max(0,line_x-x-gap - 8)
                            bbox_y1=max(0, (MAX_WORD_H+SPACE_Y)*(n_lines%max_lines) - 8 + UPPER_PADDING)
                            bbox_x2=line_x-gap + 8
                            bbox_y2=(MAX_WORD_H+SPACE_Y)*((n_lines%max_lines)+1)-SPACE_Y - (MAX_WORD_H-w_h) + 8 + UPPER_PADDING
                            t_bbox=f'{smallest_word} {bbox_x1} {bbox_y1} {bbox_x2} {bbox_y2}'
                            img_bbox.append(t_bbox)
                            continue

                        # Try adding the image without gap and break out of loop
                        elif new_x<remaining_sentence_width:
                            sorted_image_map.pop(smallest_word_index)
                            used_files.append(smallest_image)
                            t_img=cv2.resize(t_img,(new_x,w_h))
                            y,x=t_img.shape[:2]

                            # Concatenate with white space to make the height consistent
                            t_img=np.concatenate([t_img,np.ones((MAX_WORD_H-w_h,x))*255])
                            line_x+=x

                            # Append the word image and update the sentence text
                            t_sentence_img.append(t_img)
                            sentence_text+=smallest_word
                            sentence_img = np.hstack(t_sentence_img)
                            
                            # Calculate bounding box coordinates and append to the list
                            bbox_x1=max(0,line_x-x - 8)
                            bbox_y1=max(0, (MAX_WORD_H+SPACE_Y)*(n_lines%max_lines) - 8 + UPPER_PADDING)
                            bbox_x2=line_x + 8
                            bbox_y2=(MAX_WORD_H+SPACE_Y)*((n_lines%max_lines)+1)-SPACE_Y - (MAX_WORD_H-w_h) + 8 + UPPER_PADDING
                            t_bbox=f'{smallest_word} {bbox_x1} {bbox_y1} {bbox_x2} {bbox_y2}'
                            img_bbox.append(t_bbox)
                            break

                        # skip the image if it can not fit and increase the number of skipped images for while loop
                        else:
                            # print('Skipped word:',smallest_word,smallest_image)
                            skipped_images+=1
                            smallest_word_index+=1
                            # cv2.imwrite('/data/circulars/DATA/TACTFUL/skipped_images/'+smallest_image,t_img)
                            continue

                    except Exception as e:
                        # Handle exceptions and move to the next smallest word
                        print('Error',e)
                        skipped_images+=1
                        try:
                            skipped_words.append(smallest_word)
                            sorted_image_map.pop(smallest_word_index)
                            used_files.append(smallest_image)
                        except:
                            pass
                        
                # Fill the remaining space in the line with blank images
                sentence_img = np.hstack(t_sentence_img)
                residual=PAGE_W-sentence_img.shape[1]
                if residual>0:
                    sentence_img=np.hstack([sentence_img,np.ones((MAX_WORD_H,residual))*255])
                
                # Resize the sentence image to match the page width
                sentence_img=cv2.resize(sentence_img,(PAGE_W,MAX_WORD_H))
                sentence_img=np.concatenate([sentence_img,np.ones((SPACE_Y,PAGE_W))*255])
                final_image.append(sentence_img)
                n_lines+=1

                # Check if the maximum number of lines per page is reached
                if n_lines%max_lines==0:
                    saved_pages+=1
                    final_image.append(np.ones((page_left_from_bottom,PAGE_W))*255)
                    final_image.insert(0,np.ones((UPPER_PADDING,PAGE_W))*255)
                    
                    # Add space at the bottom of the page and at the top for padding
                    save_image(final_image,img_bbox,f'{language}_page_{saved_pages}',output_folder)

                    # Generate new random word height and gap
                    w_h=random.randint(MIN_WORD_H,MAX_WORD_H)
                    gap = int(w_h*weighted_avg_font_size[w_h-MIN_WORD_H]*2/3)
                    if gap<MIN_SPACE_X:
                        gap=MIN_SPACE_X
                    if gap>w_h:
                        gap=w_h
                    final_image=[]
                    img_bbox=[]
                
                y,x=img_p.shape[:2]
                data_stats[w_h] +=1
                new_x=int(w_h/y*x)
                if (int(MAX_WORD_H/3)+new_x+gap)>PAGE_W:
                    new_x=int(MIN_WORD_H/y*x)

                # Resize the image
                img=cv2.resize(img_p,(new_x,w_h))
                y,x=img.shape[:2]

                # Concatenate with white space to make the height consistent
                img=np.concatenate([img,np.ones((MAX_WORD_H-w_h,x))*255])
                line_x+=x+gap 

                sentence_img=[]
                sentence_text=' '
                sentence_img.append(np.ones((MAX_WORD_H, int(MAX_WORD_H/3)))*255)
                line_x=int(MAX_WORD_H/3)+new_x+gap

                bbox_x1=max(0,line_x-new_x-gap - 8)
                bbox_y1=max(0, (MAX_WORD_H+SPACE_Y)*(n_lines%max_lines) - 8 + UPPER_PADDING)
                bbox_x2=line_x-gap + 8
                bbox_y2=(MAX_WORD_H+SPACE_Y)*((n_lines%max_lines)+1)-SPACE_Y - (MAX_WORD_H-w_h) + 8 + UPPER_PADDING
                t_bbox=f'{image_map[img_path]} {bbox_x1} {bbox_y1} {bbox_x2} {bbox_y2}'

            else :
                # Calculate bounding box coordinates and add to the list
                bbox_x1=max(0,line_x-new_x-gap - 8)
                bbox_y1=max(0, (MAX_WORD_H+SPACE_Y)*(n_lines%max_lines) - 8 + UPPER_PADDING)
                bbox_x2=line_x-gap + 8
                bbox_y2=(MAX_WORD_H+SPACE_Y)*((n_lines%max_lines)+1)-SPACE_Y - (MAX_WORD_H-w_h) + 8 + UPPER_PADDING
                t_bbox=f'{image_map[img_path]} {bbox_x1} {bbox_y1} {bbox_x2} {bbox_y2}'

        except Exception as A:
            print('Error',A)
            continue

        # Add image and update sentence text and bounding box list
        sentence_img.append(img)
        sentence_img.append(np.ones((MAX_WORD_H, gap))*255)
        sentence_text+=image_map[img_path]
        img_bbox.append(t_bbox)

    # Save the remaining sentence if any
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
            w_h=random.randint(MIN_WORD_H,MAX_WORD_H)
            gap = int(w_h*weighted_avg_font_size[w_h-MIN_WORD_H]*2/3)
            if gap<MIN_SPACE_X:
                gap=MIN_SPACE_X
            if gap>w_h:
                gap=w_h
        except:
            pass


     # Save skipped words list to a file
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
            sorted_image_map = dict(sorted(image_map.items(), key=lambda item: len(item[1])))
            
            try:
                saved_pages = len(os.listdir(OUTPUT_FOLDER + typeset + '/images/'))
            except:
                if not os.path.exists(OUTPUT_FOLDER + typeset + '/images/'):
                    os.makedirs(OUTPUT_FOLDER + typeset + '/images/')
                    os.makedirs(OUTPUT_FOLDER + typeset + '/txt/')
                saved_pages = 0
            img=gen_images(language, input_folder + typeset + '/images/', OUTPUT_FOLDER + typeset + '/', image_map, saved_pages, sorted_image_map)
            
            
    
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