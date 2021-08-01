import math

import torch
from torch import nn
from torchvision.models.detection import (faster_rcnn, fasterrcnn_resnet50_fpn, FasterRCNN)
from torchvision.models.detection import retinanet_resnet50_fpn, retinanet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_fasterrcnn_model_for_cowboy(pretrained=True):
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    num_classes = 6
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def get_fasterrcnn_resnet153_model(num_classes, pretrained=True):
    backbone = resnet_fpn_backbone('resnet152', pretrained=pretrained, trainable_layers=5)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    return model


model_urls = {
    'retinanet_resnet50_fpn_coco': 'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth'}


def get_retinanet_model_for_cowboy():
    model = retinanet_resnet50_fpn(pretrained=True)
    num_classes = 6
    in_channels = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    model.head = retinanet.RetinaNetHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes)
    # model.head.classification_head = retinanet.RetinaNetClassificationHead(in_channels=in_channels,
    #                                                                        num_anchors=num_anchors,
    #                                                                        num_classes=num_classes)
    # model.head.regression_head = retinanet.RetinaNetRegressionHead(in_channels=in_channels,
    #                                                                num_anchors=num_anchors)

    # configure classification_head
    # prior_probability = 0.01
    # model.head.classification_head.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3,
    #                                                       stride=1, padding=1)
    # torch.nn.init.normal_(model.head.classification_head.cls_logits.weight, std=0.01)
    # torch.nn.init.constant_(model.head.classification_head.cls_logits.bias,
    #                         -math.log((1 - prior_probability) / prior_probability))
    # model.head.classification_head.num_classes = num_classes
    # model.head.classification_head.num_anchors = num_anchors

    # model.head.regression_head = retinanet.RetinaNetRegressionHead(in_channels=in_channels,
    #                                                                num_anchors=num_anchors)

    # configure regression_head

    return model


if __name__ == '__main__':
    # faster_rcnn_model = get_fasterrcnn_model_for_cowboy(pretrained=False)
    # retinanet_model_1 = get_retinanet_model_for_cowboy()
    # retinanet_model_2 = retinanet_resnet50_fpn(pretrained=True)
    rcnn_resnet152 = get_fasterrcnn_resnet153_model(6)
    # print(retinanet_model)
