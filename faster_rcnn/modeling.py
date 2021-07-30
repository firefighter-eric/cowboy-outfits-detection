import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_fasterrcnn_model_for_cowboy(pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    num_classes = 6
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == '__main__':
    model = get_fasterrcnn_model_for_cowboy(pretrained=False)
    print(model)
