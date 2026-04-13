from torchvision.models import ResNet50_Weights
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn


def create_keypointrcnn_model(num_keypoints: int = 17, pretrained: bool = True):
    if pretrained and num_keypoints == 17:
        weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        weights_backbone = None
    else:
        weights = None
        weights_backbone = ResNet50_Weights.IMAGENET1K_V1

    model = keypointrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        num_classes=2,
        num_keypoints=num_keypoints,
    )
    return model

