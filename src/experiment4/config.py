from yacs.config import CfgNode
from yacs.config import CfgNode as CN


_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = ""
_C.MODEL.SUB_NAME = ""
_C.MODEL.H_SIZE = (1000, 300)
_C.MODEL.USE_IOU = True

_C.DATASET = CN()
_C.DATASET.NAME = ""
_C.DATASET.FEAT_TYPE = "ddpn"  # ddpn / plclc
_C.DATASET.TRAIN_SIZE = 1.0

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 500
_C.TRAIN.LR = 0.01
_C.TRAIN.WEIGHT_DECAY = 0
_C.TRAIN.DEVICE = 0
_C.TRAIN.EPOCH = 5
_C.TRAIN.N_PARALLEL = 4

_C.LOG = CN()
_C.LOG.OUTDIR = "./"
_C.LOG.LOG_FILE = "log.json"
_C.LOG.NEPTUNE = False

_C.TEST = CN()
_C.TEST.CHECKPOINT = ""
_C.TEST.OUTDIR = ""
_C.TEST.DEVICE = 0


def get_cfg_defaults() -> CN:
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
