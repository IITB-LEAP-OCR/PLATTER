import os
import pandas as pd
import torch
import argparse
from tqdm import tqdm

from doctr.io import DocumentFile
from doctr.models import crnn_vgg16_bn, db_resnet50, master, parseq, sar_resnet31, vitstr_small
from doctr.models.predictor import OCRPredictor
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.datasets.vocabs import VOCABS

# If only CPU is there use cpu instead of cuda
os.environ["USE_TORCH"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def extract_number(filename):
    return int(filename.split('_')[2].split('.')[0])


def load_model(args, lang):
    
    # Detection model
    det_model = db_resnet50(pretrained=True)
    det_param = torch.load(args.det_model, map_location="cuda")
    det_model.load_state_dict(det_param)
    det_predictor = DetectionPredictor(PreProcessor((1024, 1024), batch_size=1, mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)), det_model)

    #Recognition model    
    if("crnn_vgg16_bn" in args.rec_model):
        reco_model = crnn_vgg16_bn(pretrained=False, vocab=VOCABS['iiit_' + lang])
    elif("master" in args.rec_model):
        reco_model = master(pretrained=False, vocab=VOCABS['iiit_' + lang])
    # elif(args.model=="db_mobilenet_v3_small"):
    #     reco_model = db_mobilenet_v3_small(pretrained=False, vocab=VOCABS['iiit_' + lang])
    elif("parseq" in args.rec_model):
        reco_model = parseq(pretrained=False, vocab=VOCABS['iiit_' + lang])
    elif("sar_resnet31" in args.rec_model):
        reco_model = sar_resnet31(pretrained=False, vocab=VOCABS['iiit_' + lang])
        
        
    reco_param = torch.load(args.rec_model + lang + '.pt', map_location="cuda")
    reco_model.load_state_dict(reco_param)
    reco_predictor = RecognitionPredictor(PreProcessor((32, 128), preserve_aspect_ratio=True, batch_size=1, mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301)), reco_model)        

    predictor = OCRPredictor(det_predictor, reco_predictor)
    return predictor


def get_result(input_dir, file, predictor, output_dir):
    doc = DocumentFile.from_images(input_dir + file)
    result = predictor(doc)
    json_output = result.export()
    
    predictions = []
    for page in json_output['pages']:
        dim = page['dimensions']
        for block in page['blocks']:
            for lines in block['lines']:
                for word in lines['words']:
                    values=[]
                    geo = word['geometry']
                    a = list(int(a*b) for a,b in zip(geo[0],dim))
                    b = list(int(a*b) for a,b in zip(geo[1],dim))
                    values.append(word['value'])
                    values.append(a[0])
                    values.append(a[1])
                    values.append(b[0])
                    values.append(b[1])
                    
                    predictions.append(values)
                    
                    
    predictions = sorted(predictions, key=lambda x: (x[2], x[1], x[2], x[4]))

    df = pd.DataFrame(predictions, columns=['pred', 'x0', 'y0', 'x1', 'y1'])
    df.to_csv(output_dir +  file[:-4] + '.txt', index=False, header=False, sep=' ')
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Documents OCR Input Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_file", type=str, default=None, help="path to the input folder")
    parser.add_argument("-d", "--det_model", type=str, default=None, help="Detection model directory")
    parser.add_argument("-r", "--rec_model", type=str, default=None, help="Recognition Model Directory")
    
    args = parser.parse_args()
    return args
            
if __name__ == "__main__":
    args = parse_args()
    
    languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam','odia', 'tamil', 'telugu', 'urdu']
    
    files = sorted(os.listdir(args.input_file), key=extract_number)
    
    out_dir = '/data/BADRI/OCR/results/pretrained_sar_resnet31/'
    
   
    for lang in languages:
        
        if(not os.path.exists(out_dir + lang)):
            os.makedirs(out_dir + lang)
        filtered_list = [element for element in files if element.startswith(lang)]
        predictor = load_model(args, lang)
        for file in tqdm(filtered_list):
            get_result(args.input_file, file,  predictor, out_dir + lang + '/')
