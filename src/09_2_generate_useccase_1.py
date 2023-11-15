from PIL import Image, ImageDraw, ImageFont
import pandas as pd

import fastwer

        


models = ['parseq', 'crnn_vgg16_bn', 'master', 'vitstr_small', 'crnn_mobilenet_v3_small', 'sar_resnet31']
# model = 'parseq'
for model in models:
    
    # Create a blank image with white background
    width, height = 1024, 1024
    background_color = "white"
    img = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(img)
    
    df = pd.read_csv(f'{model}.txt', sep=' ', names=['label', 'x1', 'y1', 'x2', 'y2'])
    gt_df = pd.read_csv('gt.txt', sep=' ', names=['label_gt', 'x3', 'y3', 'x4', 'y4'])
    result_df = pd.concat([df, gt_df], axis=1)
    data = result_df.values.tolist()
    # try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/lohit-punjabi/Lohit-Gurmukhi.ttf', 35) 
    for text, x1, y1, x2, y2, gt_text, x3,y3,x4,y4 in data:
        cer = fastwer.score([text], [gt_text], char_level=True)
        if(cer<4):
            draw.text((x1, y1), text, font=font, fill="black")
        else:
            draw.text((x1, y1), text, font=font, fill="red")

            

            # Save the synthetic page
    img.save(f'./images/{model}_res_2.jpg')
