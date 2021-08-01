import json
from typing import List, Dict

import torch
from torch import Tensor
from tqdm import tqdm

from data_process import CocoDataLoader
from coco_eval import coco_eval


class Pipeline:
    def __init__(self, model_checkpoint, device):
        self.model = torch.load(model_checkpoint)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, imgs: List[Tensor]):
        imgs = [_.to(self.device) for _ in imgs]
        with torch.no_grad():
            preds = self.model(imgs)
        preds = [{k: v.tolist() for k, v in pred.items()} for pred in preds]
        return preds


def preds2coco_eval_result(preds, image_ids, idx2label, threshold=0) -> List[Dict]:
    result = []
    for pred, image_id in zip(preds, image_ids):
        pred_len = len(pred['labels'])
        for i in range(pred_len):
            box, label, score = pred['boxes'][i], pred['labels'][i], pred['scores'][i]
            if score < threshold:
                continue
            bbox = box[0], box[1], box[2] - box[0], box[3] - box[1]
            result.append({'image_id': image_id,
                           'bbox': bbox,
                           'category_id': idx2label[label],
                           'score': score})
    return result


if __name__ == '__main__':
    # model_path = '../models/retinanet/m1/e9.pt'
    # out_filename = '/retinanet_m1e9.json'
    model_path = '../models/faster_rcnn/m14/e5.pt'
    out_filename = '/faster_rnn_m14e5.json'
    device = 'cuda:0'

    pipeline = Pipeline(model_path, device)

    cdl = CocoDataLoader()
    test_data = cdl.test_all
    idx2str = cdl.idx2str
    idx2label = cdl.idx2label

    preds = []
    image_ids = []

    for batched_imgs, batched_targets in tqdm(test_data):
        batched_preds = pipeline(batched_imgs)

        preds += batched_preds
        image_ids += [_[0]['image_id'] for _ in batched_targets]

    coco_result = preds2coco_eval_result(preds, image_ids, idx2label)

    with open('../outputs/eval/preds/' + out_filename, 'w') as fout:
        json.dump(preds, fout)

    with open('../outputs/eval/coco_results/' + out_filename, 'w') as fout:
        json.dump(coco_result, fout)

    coco_eval('../outputs/eval/coco_results/' + out_filename)
