import fastwer
import os
import re
import pandas as pd

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_data(file):
    f = open(file, 'r')
    result = []
    for lines in f:
        llist = lines.split(' ')
        try:
            x0, y0, x1, y1 = int(float(llist[1])), int(float(llist[2])), int(float(llist[3])), int(float(llist[4]))
            # print(llist)
        except:
            # print(type(llist[1]))
            print(llist)
            # exit()
        word = llist[0]
        bbox = [x0, y0, x1, y1]
        bboxdata = [word, bbox]
        result.append(bboxdata)
    return result


def iou(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou

# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)


def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)

results = {}
results1 = {}
# models = ['crnn_vgg16_bn', 'master', 'vitstr_small', 'crnn_mobilenet_v3_small', 'parseq']
models = ['parseq', 'crnn_vgg16_bn', 'master', 'vitstr_small', 'crnn_mobilenet_v3_small', 'sar_resnet31']
models = ['master']

for model in models:

    predictions_dir = f'/data/BADRI/OCR/results/ocr/finetuned_CHIPS1/{model}/'
    # predictions_dir = '/data/BADRI/OCR/results/ocr/gt_chips_1/parseq/'
    ground_truths_dir = '/data/BADRI/OCR/data/CHIPS1/test/txt/'

    final_predictions = []
    final_ground_truths = []
    
    lang_wise_preds = {
        'bengali': [], 'gujarati': [], 'gurumukhi': [], 'hindi': [], 'kannada': [], 'malayalam': [], 'odia': [], 'tamil': [], 'telugu': [], 'urdu': []
    }
    lang_wise_gts = {
        'bengali': [], 'gujarati': [], 'gurumukhi': [], 'hindi': [], 'kannada': [], 'malayalam': [], 'odia': [], 'tamil': [], 'telugu': [], 'urdu': []
    }

    for file in sorted(os.listdir(predictions_dir), key=natural_sort_key):
        if(file.split('_')[0]=='gujarati'):
            predictions = get_data(predictions_dir+ file)
            ground_truths = get_data(ground_truths_dir + file)

            n = len(predictions)
            for d in predictions:
                iou_max = 0
                for g in ground_truths:
                    detbox, gtbox = d[1], g[1]
                    iou_candidate = iou(gtbox, detbox)
                    if iou_candidate >= iou_max:
                        iou_max = iou_candidate
                        pred, actual = d[0], g[0]
                final_predictions.append(pred)
                final_ground_truths.append(actual)
                lang_wise_preds[file.split('_')[0]].append(pred)
                lang_wise_gts[file.split('_')[0]].append(actual)
                print(pred, actual)
        
        
    CRR = 100 - fastwer.score(final_predictions, final_ground_truths, char_level=True)
    WRR = 100 - fastwer.score(final_predictions, final_ground_truths)
    print(model,":", CRR, WRR)
    
    # #round to 2
    # CRR = round(CRR, 2)
    # WRR = round(WRR, 2)
    
    # results[model] = [CRR]
    # results1[model] = [WRR]
    
    
    # for lang in lang_wise_preds.keys():
    #     CRR = round(100 - fastwer.score(lang_wise_preds[lang], lang_wise_gts[lang], char_level=True),2)
    #     WRR = round(100 - fastwer.score(lang_wise_preds[lang], lang_wise_gts[lang]),2)
    #     results[model].append(CRR)
    #     results1[model].append(WRR)
        
        
        
       
languages = ['Overall', 'bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']


columns = []
columns1 = []
for lang in languages:
    columns.append(lang + '_CRR')
    columns1.append(lang + '_WRR')
    

df = pd.DataFrame.from_dict(results, orient='index', columns=columns)
df1 = pd.DataFrame.from_dict(results1, orient='index', columns=columns1)
transposed_df = df.transpose()
transposed_df1 = df1.transpose()
transposed_df.to_csv('crr_results.csv', index=False, sep=' ')
transposed_df1.to_csv('wrr_results.csv', index=False, sep=' ')
# df.to_csv('ocr_results.csv', index=False, sep=' ')
