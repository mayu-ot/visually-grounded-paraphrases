import pdb
from typing import Callable, Dict
import numpy as np

import chainer
import os
import json
from src.data_loader import build_dataloader, get_converter
from src.models import build_model
from yacs.config import CfgNode
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from config import get_cfg_defaults
import click


def eval_model(cfg: CfgNode) -> Dict[str, float]:

    model = build_model(cfg)
    chainer.serializers.load_npz(cfg.TEST.CHECKPOINT, model)

    converter = get_converter(
        data_name=cfg.DATASET.NAME, use_iou=cfg.MODEL.USE_IOU
    )

    val_iterator = build_dataloader("val", cfg)[0]
    test_iterator = build_dataloader("test", cfg)[0]

    device_id = cfg.TEST.DEVICE

    val_pred_scores = get_predicted_scores(
        model, val_iterator, converter, device_id
    )

    label = val_iterator.dataset.label
    precision, recall, thresholds = precision_recall_curve(
        label, val_pred_scores
    )

    f1 = 2 * (precision * recall) / (precision + recall)
    best_ind = np.nanargmax(f1)
    best_threshold = thresholds[best_ind]

    pred_scores = get_predicted_scores(
        model, test_iterator, converter, device_id
    )

    label = test_iterator.dataset.label
    pred_label = pred_scores > best_threshold

    f1 = f1_score(label, pred_label)
    prec = precision_score(label, pred_label)
    recall = recall_score(label, pred_label)

    return {"f1": f1, "precision": prec, "recall": recall}


def get_predicted_scores(
    model: chainer.Chain,
    test_iterator: chainer.iterators.MultiprocessIterator,
    converter: Callable,
    device_id: int,
) -> np.ndarray:

    device = chainer.get_device(device_id)
    device.use()

    model.to_gpu()

    pred_scores = []

    with chainer.using_config("train", False):
        with chainer.using_config("enable_backprop", False):
            for batch in test_iterator:
                inputs = converter(batch, device)
                y = model.predict(*inputs)
                y.to_cpu()
                pred_scores.append(y.data.ravel())

    return np.hstack(pred_scores)


@click.command()
@click.argument("config_file")
def main(config_file: str):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.TRAIN.N_PARALLEL = 1
    cfg.freeze()
    metrics = eval_model(cfg)

    out_file = os.path.join(cfg.TEST.OUTDIR, "performance_score.json")
    json.dump(metrics, open(out_file, "w"))

    for k, v in metrics.items():
        print(f"{k}: {v:.2}")


if __name__ == "__main__":
    main()
