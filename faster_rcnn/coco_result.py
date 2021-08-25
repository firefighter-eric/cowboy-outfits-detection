import json

import cv2
import pandas as pd
from torchvision.transforms import ToTensor
from tqdm import tqdm

from configuration import Args
from data_process import CocoDataLoader
from evaluate import Pipeline, preds2coco_eval_result
from plot import plot

if __name__ == '__main__':
    args = Args()
    img_path = '../data/images/'

    cdl = CocoDataLoader(args.data_dir, from_cache=True)
    idx2str = cdl.idx2str
    idx2label = cdl.idx2label

    # df = pd.read_csv('../data/valid.csv')
    df = pd.read_csv('../data/test.csv')
    image_ids = df.id.to_list()
    filenames = df.file_name.to_list()

    pipeline = Pipeline(args.best_model_path, args.device)
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

    threshold = 0
    coco_result = preds2coco_eval_result(preds, image_ids, idx2label, threshold=threshold)

    # with open('../outputs/preds/eval.json', 'w') as fout:
    #     json.dump(preds, fout)

    with open(args.coco_result_path, 'w') as fout:
        json.dump(coco_result, fout)

    for img, pred in zip(imgs, preds):
        plot(img=img, target=pred, id2str=idx2str, threshold=threshold)
