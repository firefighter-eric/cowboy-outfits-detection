from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

cocoGt = COCO('../data/train.json')
cocoDt = cocoGt.loadRes('../outputs/coco_results/m8e9.json')
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
