from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_eval(res_path):
    coco_gt = COCO('../data/train.json')
    coco_dt = coco_gt.loadRes(res_path)
    ce = COCOeval(coco_gt, coco_dt, "bbox")
    ce.evaluate()
    ce.accumulate()
    ce.summarize()
