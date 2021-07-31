from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, faster_rcnn


def get_fasterrcnn_model_for_cowboy(pretrained=True):
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    num_classes = 6
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def get_retinanet_model_for_cowboy(pretrained=True):
    model = retinanet_resnet50_fpn(pretrained=pretrained, num_classes=6)
    return model


if __name__ == '__main__':
    faster_rcnn_model = get_fasterrcnn_model_for_cowboy(pretrained=False)
    retinanet_model = get_retinanet_model_for_cowboy(pretrained=False)
    print(retinanet_model)
