from typing import Dict, List, Optional, Callable, Union
from src.models import build_model
from src.data_loader import build_dataloader, get_converter

import chainer
from chainer import training
from chainer.training import extensions
import os
import json
from config import get_cfg_defaults
from yacs.config import CfgNode
import click
import socket
import getpass
import sys
import neptune


def log2neptune(trainer, logfile: str) -> None:
    filename = os.path.join(trainer.out, logfile)
    log_info = json.load(open(filename, "r"))
    log_info = log_info[-1]
    iteration = log_info["iteration"]
    for k, v in log_info.items():
        neptune.log_metric(k, iteration, v)


def setup_neptune(cfg) -> None:
    neptune.init(project_qualified_name="mayu-ot/VGP")
    neptune.create_experiment(
        name=f"train {cfg.MODEL.GATE}",
        properties={
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "wd": os.getcwd(),
            "cmd": " ".join(sys.argv),
        },
        tags=["train"],
    )
    filename = os.path.join(cfg.LOG.OUTDIR, "config.yaml")
    neptune.log_artifact(filename, "config.yaml")


def train(
    cfg: CfgNode,
    checkpoint_on: bool = True,
    my_extensions: Optional[List[Callable]] = None,
) -> Dict[str, Union[str, float]]:

    model = build_model(cfg)

    opt = chainer.optimizers.Adam(cfg.TRAIN.LR)
    opt.setup(model)

    if cfg.TRAIN.WEIGHT_DECAY is not None:
        opt.add_hook(
            chainer.optimizer.WeightDecay(cfg.TRAIN.WEIGHT_DECAY), "hook_dec"
        )

    converter = get_converter(
        data_name=cfg.DATASET.NAME, use_iou=cfg.MODEL.USE_IOU
    )

    train_iterators = build_dataloader("train", cfg)

    if len(train_iterators) > 1:
        updater = training.updaters.MultiprocessParallelUpdater(
            train_iterators,
            opt,
            converter=converter,
            devices={"main": 0, "second": 1, "third": 2, "fourth": 3},
        )
    else:
        device0 = chainer.get_device(cfg.TRAIN.DEVICE)
        device0.use()
        updater = training.StandardUpdater(
            train_iterators[0], opt, converter=converter, device=device0
        )

    trainer = training.Trainer(
        updater, (cfg.TRAIN.EPOCH, "epoch"), cfg.LOG.OUTDIR
    )

    val_interval = (1, "epoch")
    log_interval = (10, "iteration")

    val_iterator = build_dataloader("val", cfg)[0]

    trainer.extend(
        extensions.Evaluator(
            val_iterator, model, converter=converter, device=cfg.TRAIN.DEVICE
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
            extensions.snapshot_object(model, "model"), trigger=best_val_trigger
        )

    # logging extensions
    log_report = extensions.LogReport(
        trigger=log_interval, filename=cfg.LOG.LOG_FILE
    )
    trainer.extend(log_report)

    if cfg.LOG.NEPTUNE:
        trainer.extend(
            lambda trainer: log2neptune(trainer, cfg.LOG.LOG_FILE),
            trigger=log_interval,
        )

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
            os.path.join(cfg.LOG.OUTDIR, "final_model"), model
        )

    best_val = max(
        [
            log["validation/main/f1"]
            for log in log_report.log
            if "validation/main/f1" in log
        ]
    )

    result_info = {"log_dir": cfg.LOG.OUTDIR, "best_val": best_val}
    return result_info


@click.command()
@click.argument("config_file")
def main(config_file: str):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    if not os.path.exists(cfg.LOG.OUTDIR):
        os.makedirs(cfg.LOG.OUTDIR)
        filename = os.path.join(cfg.LOG.OUTDIR, "config.yaml")
        with open(filename, "w") as f:
            f.write(cfg.dump())

    if cfg.LOG.NEPTUNE:
        setup_neptune(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
