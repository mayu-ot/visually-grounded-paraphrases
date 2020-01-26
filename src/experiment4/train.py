import pdb
from typing import List, Optional, Tuple, Callable
from src.models.models import iParaphraseNet
from src.data_loader.multimodal_dataloader import (
    MultiModalDataLoader,
    DownSampler,
)
from chainer.dataset.convert import to_device
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
import os
from chainer.iterators import SerialIterator
import json
from datetime import datetime as dt
from optuna.integration import ChainerPruningExtension


def custom_converter(batch, device=None):
    phrase_a = [to_device(device, np.asarray(b[0], np.int32)) for b in batch]
    phrase_b = [to_device(device, np.asarray(b[1], np.int32)) for b in batch]

    visfeat_a = np.vstack([b[2] for b in batch])
    visfeat_a = to_device(device, visfeat_a)
    visfeat_b = np.vstack([b[3] for b in batch])
    visfeat_b = to_device(device, visfeat_b)

    label = np.asarray([b[4] for b in batch], np.int32)
    label = to_device(device, label)
    return phrase_a, phrase_b, visfeat_a, visfeat_b, label


def construct_model(gate_mode: str, h_size: Tuple[int, int] = (1000, 300)):
    model = iParaphraseNet(gate_mode, h_size)
    return model


def get_dataset(splits: List[str], subset_size):
    feat_type: str = "ddpn"
    data = []
    for s in splits:
        dataset_file = f"data/processed/{feat_type}/{s}.csv"
        d = MultiModalDataLoader(dataset_file, s, subset_size, feat_type)
        data.append(d)

    return data


def prepare_logging_directory(out_pref: str = "./") -> str:
    time_stamp = dt.now().strftime("%Y%m%d-%H%M%S")
    saveto = os.path.join(out_pref, time_stamp)
    os.makedirs(saveto)
    return saveto


def train(
    model: chainer.Chain,
    train_iterator: chainer.iterators.SerialIterator,
    val_iterator: chainer.iterators.SerialIterator,
    epoch: int,
    device: int,
    lr: float,
    w_decay: Optional[float],
    out_pref: str,
    checkpoint_on: bool = True,
    my_extensions: Optional[List[Callable]] = None,
    data_parallel: bool = True,
) -> dict:

    saveto = prepare_logging_directory(out_pref)

    # args = locals()
    # json.dump(args, open(os.path.join(saveto, "args"), "w"))

    device0 = chainer.get_device(device)
    device0.use()

    opt = chainer.optimizers.Adam(lr)
    opt.setup(model)

    if w_decay is not None:
        opt.add_hook(chainer.optimizer.WeightDecay(w_decay), "hook_dec")

    if data_parallel:
        updater = training.updaters.ParallelUpdater(
            train_iterator,
            opt,
            converter=custom_converter,
            devices={"main": 0, "second": 1, "third": 2},
        )
    else:
        updater = training.StandardUpdater(
            train_iterator, opt, converter=custom_converter, device=device0
        )
    trainer = training.Trainer(updater, (epoch, "epoch"), saveto)

    val_interval = (1, "epoch")
    log_interval = (10, "iteration")

    trainer.extend(
        extensions.Evaluator(
            val_iterator, model, converter=custom_converter, device=device
        ),
        trigger=val_interval,
    )

    trainer.extend(
        extensions.ExponentialShift("alpha", 0.1), trigger=(1, "epoch")
    )

    if checkpoint_on:
        best_val_trigger = training.triggers.MaxValueTrigger(
            "validation/main/f1", trigger=val_interval
        )

        trainer.extend(
            extensions.snapshot_object(model, "model"),
            trigger=best_val_trigger,
        )

    # logging extensions
    log_report = extensions.LogReport(trigger=log_interval)
    trainer.extend(log_report)
    trainer.extend(extensions.observe_lr(), trigger=log_interval)

    trainer.extend(
        extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "main/loss",
                "main/f1",
                "validation/main/loss",
                "validation/main/f1",
            ]
        ),
        trigger=log_interval,
    )

    trainer.extend(extensions.ProgressBar(update_interval=log_interval[0]))

    if my_extensions is not None:
        for ext in my_extensions:
            trainer.extend(ext)

    print("start training")
    trainer.run()

    if checkpoint_on:
        chainer.serializers.save_npz(
            os.path.join(saveto, "final_model"), model
        )

    best_val = max(
        [
            log["validation/main/f1"]
            for log in log_report.log
            if "validation/main/f1" in log
        ]
    )

    result_info = {"log_dir": saveto, "best_val": best_val}
    return result_info


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="training script for a paraphrase classifier"
    )
    parser.add_argument(
        "gate_mode",
        choices=["none", "language_gate", "visual_gate", "multimodal_gate"],
    )

    parser.add_argument(
        "--lr", "-lr", type=float, default=0.01, help="learning rate <float>"
    )
    parser.add_argument(
        "--device", "-d", type=int, default=None, help="gpu device id <int>"
    )

    parser.add_argument(
        "--b_size",
        "-b",
        type=int,
        default=500,
        help="minibatch size <int> (default 500)",
    )
    parser.add_argument(
        "--epoch", "-e", type=int, default=5, help="maximum epoch <int>"
    )
    parser.add_argument(
        "--san_check", "-sc", action="store_true", help="sanity check mode"
    )
    parser.add_argument(
        "--w_decay",
        "-wd",
        type=float,
        default=None,
        help="weight decay <float>",
    )
    parser.add_argument(
        "--train_percent",
        type=float,
        default=1.0,
        help="How many date will be used for training",
    )
    parser.add_argument("--out_pref", type=str, default="./checkpoint/")
    args = parser.parse_args()

    train_dataset, val_dataset = get_dataset(["train", "val"])

    down_sampler = DownSampler(
        train_dataset.indices_positive,
        train_dataset.indices_negative,
        args.train_percent,
    )

    train_iterator = SerialIterator(
        train_dataset, args.b_size, shuffle=None, order_sampler=down_sampler
    )

    val_iterator = SerialIterator(val_dataset, 2 * args.b_size, repeat=False)

    model = construct_model(args.gate_mode)
    if args.device is not None:
        chainer.cuda.get_device_from_id(args.device).use()
        model.to_gpu()

    train(
        model,
        train_iterator,
        val_iterator,
        args.epoch,
        args.device,
        args.lr,
        args.w_decay,
        args.out_pref,
    )


if __name__ == "__main__":
    main()
