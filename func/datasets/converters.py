import numpy as np
import random
from chainer.dataset.convert import to_device

def jitter_bbox(bbox, im_sizes, aspect_band=4/5, offset_band=0.1):
    '''
    bbox: numpy array of bounding boxes. bbox[i] is (ymin, xmin, ymax, xmax)
    im_sizes: array of image sizes. im_size[i] is (channel, height, width)
    '''
    if not isinstance(bbox, np.ndarray):
        bbox = np.asarray(bbox)
    
    if bbox.dtype != 'float32':
        bbox = bbox.astype('f')
        
    _bbox = bbox.copy()
    y_min = _bbox[:, 0]
    x_min = _bbox[:, 1]
    
    w = _bbox[:, -1] - _bbox[:, 1]
    h = _bbox[:, 2] - _bbox[:, 0]
    
    n = len(_bbox)
    
    w_ratio = np.asarray([random.uniform(aspect_band, 1/aspect_band) for _ in range(n)])
    h_ratio = np.asarray([random.uniform(aspect_band, 1/aspect_band) for _ in range(n)])
    w *= w_ratio
    h *= h_ratio
    
    x_offset = w*np.asarray([random.uniform(-offset_band, offset_band) for _ in range(w.size)])
    y_offset = h*np.asarray([random.uniform(-offset_band, offset_band) for _ in range(w.size)])
    
    x_min += np.fmax(x_offset, 0) # clip at 0
    y_min += np.fmax(y_offset, 0) # clip at 0
    y_max = np.clip(y_min + h, 0, im_sizes[:, 1])
    x_max = np.clip(x_min + w, 0, im_sizes[:, 2])
    
    j_bbox = np.zeros_like(bbox)
    j_bbox[:, 0] = y_min
    j_bbox[:, 1] = x_min
    j_bbox[:, 2] = y_max
    j_bbox[:, 3] = x_max
    
    return j_bbox

def cvrt_bbox(batch, device=None, aspect_band=2/3, offset_band=0.4):
    im = [b[0].transpose(2, 0, 1).astype('f') for b in batch]
    im_shape = np.asarray([x.shape for x in im])
    roi = [b[1] for b in batch]
    j_rois = jitter_bbox(roi, im_shape, aspect_band, offset_band)
    
    if device is not None:
        im = [to_device(device, x) for x in im]
        j_rois = to_device(device, j_rois)
    return im, j_rois

def cvrt_frcnn_input(batch, device=None):
    im = [b[0].transpose(2, 0, 1).astype('f') for b in batch]
    roi = [b[1] for b in batch]
    roi = np.vstack(roi)
    
    if device is not None:
        im = [to_device(device, x) for x in im]
        roi = to_device(device, roi)
        
    return im, roi

def cvrt_pre_comp_feat(batch, device=None):
    phr_1 = np.vstack([b[0] for b in batch]).astype('f')
    phr_2 = np.vstack([b[1] for b in batch]).astype('f')
    xvis_1 = np.vstack([b[2] for b in batch]).astype('f')
    xvis_2 = np.vstack([b[3] for b in batch]).astype('f')
    l = np.asarray([b[4] for b in batch]).astype('i')
    
    if device is not None:
        phr_1 = to_device(device, phr_1)
        phr_2 = to_device(device, phr_2)
        xvis_1 = to_device(device, xvis_1)
        xvis_2 = to_device(device, xvis_2)
        l = to_device(device, l)
    
    return phr_1, phr_2, xvis_1, xvis_2, l