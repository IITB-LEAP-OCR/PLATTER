import argparse
import os


def process_hocr(input_file,output_folder):

    with open(input_file,'r',encoding='utf-8') as f:
        bboxes=f.read().split('\n')
    # print(bboxes)
    hocr=''
    prev_bbox_y=0
    lines=[]
    temp_line=[]
    for bbox in bboxes:
        if len(bbox)>0:
            text,xl,yl,xh,yh=bbox.split()
            if abs(int(yl)-prev_bbox_y)>7:
                lines.append(temp_line)
                temp_line=[]
            temp_line.append(bbox)
            prev_bbox_y=int(yl)
    if len(temp_line)>0:
        lines.append(temp_line)
    # print(lines)
    for line in lines:
        _,xl,yl,_,_=line[0].split()
        _,_,_,xh,yh=line[-1].split()
        title=f'{xl} {yl} {xh} {yh}'
        span=f'\t<span class="ocr_line" title="bbox {title}" style="position:absolute; top:{yl}px ; left:{xl}px;">\n'
        for word in line:
            span+=f'\t\t<span class="ocrx_word" title="{xl} {yl} {xh} {yh}" style="display:inline-block;font_size=2em;">{text}</span>\n'
        span+='\t</span>\n'
        hocr+=span 
        
    html=f'''
    <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{input_file.split('/')[-1].split('.')[0]}</title>
        </head>
        <body>
    {hocr}
        </body>
    </html>'''

    print(os.path.join(output_folder,f"{input_file.split('/')[-1][:-4]}_hocr_output.html"))
    with open(os.path.join(output_folder,f"{input_file.split('/')[-1][:-4]}_hocr_output.html"),'w') as f:
        f.write(html)

def main(args):
    if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            
    process_hocr(args.input_file,args.output_folder)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_file", type=str, default=None, help="path to the input img file")
    parser.add_argument("-o", "--output_folder", type=str, default="./", help="path to the output img directory")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)