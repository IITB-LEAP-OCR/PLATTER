import fastwer
import os

def get_data(file):
    f = open(file, 'r')
    result = []
    for lines in f:
        llist = lines.split(' ')
        x0, y0, x1, y1 = int(llist[1]), int(llist[2]), int(llist[3]), int(llist[4])
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


models = ['crnn_vgg16_bn', 'master', 'vitstr_small', 'crnn_mobilenet_v3_small', 'parseq']

for model in models:

    predictions_dir = f'/data/BADRI/OCR/results/ocr/finetuned_CHIPS_1/{model}/'
    predictions_dir = '/data/BADRI/OCR/results/ocr/gt_chips_1/parseq/'
    ground_truths_dir = '/data/BADRI/OCR/data/CHIPS_1/test/txt/'

    final_predictions = []
    final_ground_truths = []

    for file in os.listdir(predictions_dir):
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
        
        
    CRR = 100 - fastwer.score(final_predictions, final_ground_truths, char_level=True)
    WRR = 100 - fastwer.score(final_predictions, final_ground_truths)
    print(model,":", CRR, WRR)