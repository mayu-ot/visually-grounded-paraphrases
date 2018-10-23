from chainercv.links.model.faster_rcnn import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn.faster_rcnn_vgg import _roi_pooling_2d_yx
import numpy as np
import chainer
import chainer.functions as F


class FasterRCNNExtractor(FasterRCNNVGG16):
    """docstring for FasterRCNNExtractor"""

    def extract_head(self, x, rois, roi_indices):
        head = self.head
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, head.roi_size, head.roi_size,
            head.spatial_scale)

        fc6 = F.relu(head.fc6(pool))
        fc7 = F.relu(head.fc7(fc6))
        return fc7

    def extract(self, x, bbox, roi_indices):
        '''
        x (array): Demension of each image array is CxHxW (RGB).
                   Image array after prepare()
        bbox (array): bbox[i] contain roi coordinates of i-th image.
                      Each item is (y_min, x_min, y_max, x_max).
        roi_indices: indices to each roi to image
        '''

        h = self.extractor(x)
        feat = self.extract_head(h, bbox, roi_indices)
        return feat