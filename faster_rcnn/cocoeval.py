import json

import cv2
import pandas as pd
from torchvision.transforms import ToTensor
from tqdm import tqdm
from plot import plot

from data_process import CocoDataLoader
from evaluate import Pipeline, preds2coco_eval_result

if __name__ == '__main__':
    img_path = '../data/images/'
    model_path = '../outputs/models/m7/m5.pt'
    device = 'cuda:0'

    cdl = CocoDataLoader()
    idx2str = cdl.idx2str
    idx2label = cdl.idx2label

    df = pd.read_csv('../data/valid.csv')
    image_ids = df.id.to_list()
    filenames = df.file_name.to_list()

    pipeline = Pipeline(model_path, device)
    transform = ToTensor()

    preds = []
    imgs = []
    for filename in tqdm(filenames):
        img = cv2.imread(img_path + filename, 3)
        img = transform(img)
        # BGR -> RGB
        img = img[[2, 1, 0]]
        preds += pipeline([img])
        imgs.append(img)

    threshold = 0.1
    coco_result = preds2coco_eval_result(preds, image_ids, idx2label, threshold=threshold)

    # with open('../outputs/preds/eval.json', 'w') as fout:
    #     json.dump(preds, fout)

    with open('../outputs/coco_results/m7e5.json', 'w') as fout:
        json.dump(coco_result, fout)

    for img, pred in zip(imgs, preds):
        plot(img=img, target=pred, id2str=idx2str, threshold=threshold)
