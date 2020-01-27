import pdb
from typing import List, Optional, Tuple, Callable
import numpy as np

import chainer
import os
from chainer.iterators import SerialIterator
import json
from train import get_dataset, construct_model, custom_converter
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)


def evaluation(config_json: str, model_ckpt: str, device: int = 0):
    val_dataset, test_dataset = get_dataset(["val", "test"])
    val_iterator = SerialIterator(
        val_dataset, 1000, repeat=False, shuffle=False
    )
    test_iterator = SerialIterator(
        test_dataset, 1000, repeat=False, shuffle=False
    )

    val_pred_scores = get_predicted_scores(
        config_json, model_ckpt, val_iterator, device=device
    )

    label = val_dataset.label
    precision, recall, thresholds = precision_recall_curve(
        label, val_pred_scores
    )

    f1 = 2 * (precision * recall) / (precision + recall)
    best_ind = np.nanargmax(f1)
    best_threshold = thresholds[best_ind]

    pred_scores = get_predicted_scores(
        config_json, model_ckpt, test_iterator, device=device
    )

    label = test_dataset.label
    pred_label = pred_scores > best_threshold

    f1 = f1_score(label, pred_label)
    prec = precision_score(label, pred_label)
    recall = recall_score(label, pred_label)

    out_dir = os.path.dirname(model_ckpt)
    out_file = os.path.join(out_dir, "performance_score.json")
    json.dump(
        {"f1": f1, "precision": prec, "recall": recall}, open(out_file, "w")
    )

    print(f"f1: {f1:.2}, prec: {prec:.2} recall: {recall:.2}")


def get_predicted_scores(
    config_json: str,
    model_ckpt: str,
    test_iterator: chainer.iterators.SerialIterator,
    device: int,
) -> np.ndarray:

    config = json.load(open(config_json))

    model = construct_model(
        config["gate_mode"], (config["h_size_0"], config["h_size_1"])
    )

    chainer.serializers.load_npz(model_ckpt, model)

    device0 = chainer.get_device(device)
    device0.use()

    model.to_gpu()

    print("start evaluation")
    pred_scores = []

    with chainer.using_config("train", False):
        with chainer.using_config("enable_backprop", False):
            for batch in test_iterator:
                inputs = custom_converter(batch, device0)
                y = model.predict(*inputs)
                y.to_cpu()
                pred_scores.append(y.data.ravel())

    return np.hstack(pred_scores)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="evaluation script for a paraphrase classifier"
    )
    parser.add_argument("config_json", type=str)
    parser.add_argument("model_ckpt", type=str)
    parser.add_argument(
        "--device", "-d", type=int, default=None, help="gpu device id <int>"
    )
    args = parser.parse_args()

    evaluation(args.config_json, args.model_ckpt, args.device)


if __name__ == "__main__":
    main()
