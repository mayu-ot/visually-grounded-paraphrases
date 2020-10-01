import pdb
from typing import Dict, List, Optional, Callable, Union
from src.models import build_model
from src.data_loader import build_dataloader, get_converter
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
import os
import tempfile
import joblib
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


def multimodal_iou_converter(batch, device=None):
    phrase_a = [to_device(device, np.asarray(b[0], np.int32)) for b in batch]
    phrase_b = [to_device(device, np.asarray(b[1], np.int32)) for b in batch]

    visfeat_a = np.vstack([b[2] for b in batch])
    visfeat_a = to_device(device, visfeat_a)
    visfeat_b = np.vstack([b[3] for b in batch])
    visfeat_b = to_device(device, visfeat_b)

    ious = np.asarray([b[4] for b in batch], np.float32)[:, None]
    ious = to_device(device, ious)

    label = np.asarray([b[5] for b in batch], np.int32)
    label = to_device(device, label)
    return phrase_a, phrase_b, visfeat_a, visfeat_b, ious, label


def multimodal_converter(batch, device=None):
    phrase_a = [to_device(device, np.asarray(b[0], np.int32)) for b in batch]
    phrase_b = [to_device(device, np.asarray(b[1], np.int32)) for b in batch]

    visfeat_a = np.vstack([b[2] for b in batch])
    visfeat_a = to_device(device, visfeat_a)
    visfeat_b = np.vstack([b[3] for b in batch])
    visfeat_b = to_device(device, visfeat_b)

    label = np.asarray([b[4] for b in batch], np.int32)
    label = to_device(device, label)
    return phrase_a, phrase_b, visfeat_a, visfeat_b, label


def phrase_converter(batch, device):
    phrase_a = [to_device(device, np.asarray(b[0], np.int32)) for b in batch]
    phrase_b = [to_device(device, np.asarray(b[1], np.int32)) for b in batch]
    label = np.asarray([b[2] for b in batch], np.int32)
    label = to_device(device, label)
    return phrase_a, phrase_b, label


# def get_converter(data_name: str, use_iou: bool = False) -> Callable:
#     if data_name == "multimodal":
#         if use_iou:
#             cnv_f = multimodal_iou_converter
#         else:
#             cnv_f = multimodal_converter
#     elif data_name == "phrase-only":
#         cnv_f = phrase_converter
#     return cnv_f


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
            extensions.snapshot_object(model, "model"),
            trigger=best_val_trigger,
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


# def load_study_log_from_neptune(exp_id: str) -> dict:
#     project = neptune.init(project_qualified_name="mayu-ot/VGP")
#     neptune_exp = project.get_experiments(exp_id)[0]
#     with tempfile.TemporaryDirectory() as d:
#         neptune_exp.download_artifact("study.pkl", destination_dir=d)
#         file_name = os.path.join(d, "study.pkl")
#         study = joblib.load(open(file_name, "rb"))
#     return study.best_params


# def train_model_with_study_log(args):
#     exp_id: "str" = args.exp_id

#     project = neptune.init(project_qualified_name="mayu-ot/VGP")
#     neptune_exp = project.get_experiments(exp_id)[0]

#     params: dict = load_study_log_from_neptune(exp_id)

#     if params["weight_decay_on"]:
#         weight_decay: Optional[float] = params["weight_decay"]
#     else:
#         weight_decay = None

#     exp_name = neptune_exp.name.split("_")
#     gate_mode = "_".join(exp_name[2:])
#     params["gate_mode"] = gate_mode

#     if "h_size_0" in params:
#         h_size_0 = params["h_size_0"]
#         h_size_1 = params["h_size_1"]
#     else:
#         h_size_0 = 1000
#         h_size_1 = 300
#     model = build_model(gate_mode, (h_size_0, h_size_1))

#     train_dataset, val_dataset = get_dataset(["train", "val"])

#     down_sampler = DownSampler(
#         train_dataset.indices_positive, train_dataset.indices_negative
#     )

#     b_size: int = 1000

#     train_iterator = SerialIterator(
#         train_dataset, b_size, shuffle=None, order_sampler=down_sampler
#     )

#     val_iterator = SerialIterator(val_dataset, 2 * b_size, repeat=False)

#     result_info = train(
#         model,
#         train_iterator,
#         val_iterator,
#         5,
#         args.device,
#         params["lr"],
#         weight_decay,
#         out_pref=args.out_pref,
#         checkpoint_on=True,
#         data_parallel=False,
#     )

#     if cfg.LOG.NEPTUNE:
#         neptune.stop()

#     param_file = os.path.join(result_info["log_dir"], "config.json")
#     json.dump(params, open(param_file, "w"))


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
