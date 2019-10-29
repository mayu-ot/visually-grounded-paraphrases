from  torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FasterRCNNExtractor(FasterRCNN):
    """docstring for FasterRCNNExtractor"""

    def extract_heads(self, features, bboxes, image_shapes):
        head = self.roi_heads
        box_features = self.box_roi_pool(features, bboxes, image_shapes)
        box_features = self.box_head(box_features)
        return box_features

    def extract(self, images, bboxes):
        """
        images (list[Tensor]): images to be processed
        bboxes (list[Tensor]): list of bounding boxes [x1, y1, x2, y2].
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        box_features = self.extract_heads(features, bboxes, images.image_sizes)

        return box_features
    
def resnet_frcnn_model(pretrained, num_classes=91):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', False)
    model = FasterRCNNExtractor(backbone, num_classes)
    return model