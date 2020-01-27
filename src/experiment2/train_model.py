from datetime import datetime as dt
import json
from chainer.iterators import SerialIterator, MultiprocessIterator
import os
import numpy as np
import chainer
from chainer.training import extensions
from chainer import training
from chainer.dataset.convert import to_device

from src.models.phrase_only_models import (
    PhraseOnlyNet,
    WordEmbeddingAverage,
    LSTMPhraseEmbedding,
)
from src.data_loader.phrase_only_dataloader import PhraseOnlyDataLoader
import sys

sys.path.append("./")


chainer.config.multiproc = False  # single proc is faster


def custom_converter(batch, device):
    phrase_a = [to_device(device, np.asarray(b[0], np.int32)) for b in batch]
    phrase_b = [to_device(device, np.asarray(b[1], np.int32)) for b in batch]
    label = np.asarray([b[2] for b in batch], np.int32)
    label = to_device(device, label)
    return phrase_a, phrase_b, label


def get_model(model_type):

    if model_type == "wea":
        phrase_emb = WordEmbeddingAverage()
    elif model_type == "lstm":
        phrase_emb = LSTMPhraseEmbedding(300)
    else:
        raise RuntimeError("invalid model type")

    model = PhraseOnlyNet(phrase_emb)

    return model


def get_data(split, san_check):
    return PhraseOnlyDataLoader(split, san_check)


def train(
    san_check=False,
    epoch=5,
    lr=0.001,
    b_size=500,
    device=0,
    w_decay=None,
    out_pref="./checkpoints/",
    model_type="vis+lng",
    pl_type=None,
    gate_mode=None,
):
    args = locals()

    time_stamp = dt.now().strftime("%Y%m%d-%H%M%S")
    saveto = out_pref + "sc_" * san_check + time_stamp + "/"
    os.makedirs(saveto)
    json.dump(args, open(saveto + "args", "w"))
    print("output to", saveto)
    print("setup dataset...")

    train = get_data("train", san_check)
    val = get_data("val", san_check)

    if chainer.config.multiproc:
        train_iter = MultiprocessIterator(train, b_size, n_processes=2)
        val_iter = MultiprocessIterator(
            val, b_size, shuffle=False, repeat=False, n_processes=2
        )
    else:
        train_iter = SerialIterator(train, b_size)
        val_iter = SerialIterator(val, b_size * 2, shuffle=False, repeat=False)

    print("setup a model: %s" % model_type)

    model = get_model(model_type)

    opt = chainer.optimizers.Adam(lr)
    opt.setup(model)

    if device is not None:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    if w_decay:
        opt.add_hook(chainer.optimizer.WeightDecay(w_decay), "hook_dec")

    updater = training.StandardUpdater(
        train_iter, opt, converter=custom_converter, device=device
    )

    trainer = training.Trainer(updater, (epoch, "epoch"), saveto)

    val_interval = (1, "epoch") if san_check else (1000, "iteration")
    log_interval = (1, "iteration") if san_check else (10, "iteration")
    prog_interval = 1 if san_check else 10

    trainer.extend(
        extensions.Evaluator(
            val_iter, model, converter=custom_converter, device=device
        ),
        trigger=val_interval,
    )

    if not san_check:
        trainer.extend(extensions.ExponentialShift("alpha", 0.1), trigger=(1, "epoch"))

    # # Comment out to enable visualization of a computational graph.
    trainer.extend(extensions.dump_graph("main/loss"))
    if not san_check:
        """
        Comment out next line to save a checkpoint at each epoch, which enable you
        to restart training loop from the saved point. Note that saving a checkpoint
        may cost a few minutes.
        """
        trainer.extend(extensions.snapshot(), trigger=(1, "epoch"))

        best_val_trigger = training.triggers.MaxValueTrigger(
            "validation/main/f1", trigger=val_interval
        )

        trainer.extend(
            extensions.snapshot_object(model, "model"), trigger=best_val_trigger
        )

    # logging extensions
    trainer.extend(extensions.LogReport(trigger=log_interval))

    trainer.extend(extensions.observe_lr(), trigger=log_interval)

    trainer.extend(
        extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "main/loss",
                "main/f1",
                "main/validation/loss",
                "validation/main/f1",
            ]
        ),
        trigger=log_interval,
    )

    trainer.extend(extensions.ProgressBar(update_interval=prog_interval))

    print("start training")
    trainer.run()

    chainer.serializers.save_npz(saveto + "final_model", model)

    if not san_check:
        return best_val_trigger._best_value


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="training script for a paraphrase classifier"
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
        "--w_decay", "-wd", type=float, default=None, help="weight decay <float>"
    )
    parser.add_argument("--model_type", "-mt", default="wea")
    parser.add_argument("--out_pref", type=str, default="./checkpoint/")
    args = parser.parse_args()

    args_dic = vars(args)

    train(
        san_check=args_dic["san_check"],
        epoch=args_dic["epoch"],
        lr=args_dic["lr"],
        b_size=args_dic["b_size"],
        device=args_dic["device"],
        w_decay=args_dic["w_decay"],
        model_type=args_dic["model_type"],
        out_pref=args_dic["out_pref"],
    )


if __name__ == "__main__":
    main()