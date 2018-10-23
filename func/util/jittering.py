import numpy as np
import random

def jitter_bbox(bbox, im_sizes):
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
    
    w_ratio = np.asarray([random.uniform(4/5, 5/4) for _ in range(n)])
    h_ratio = np.asarray([random.uniform(4/5, 5/4) for _ in range(n)])
    w *= w_ratio
    h *= h_ratio
    
    x_offset = w*np.asarray([random.uniform(-0.1, 0.1) for _ in range(w.size)])
    y_offset = h*np.asarray([random.uniform(-0.1, 0.1) for _ in range(w.size)])
    
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