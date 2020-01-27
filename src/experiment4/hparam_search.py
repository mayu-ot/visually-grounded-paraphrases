import optuna

# from optuna.integration import
from train import get_dataset, construct_model, train
from src.data_loader.multimodal_dataloader import DownSampler
from chainer.iterators import SerialIterator
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import socket
import getpass
import os
import sys
from optuna.integration import ChainerPruningExtension


def objective(
    trial: optuna.trial.Trial,
    gate_mode: str,
    train_iterator: SerialIterator,
    val_iterator: SerialIterator,
    device: int = 0,
):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay_on = trial.suggest_categorical(
        "weight_decay_on", [True, False]
    )
    if weight_decay_on:
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    else:
        weight_decay = None

    h_size_0 = trial.suggest_int("h_size_0", 100, 3000)
    h_size_1 = trial.suggest_int("h_size_1", 100, 3000)

    model = construct_model(gate_mode, (h_size_0, h_size_1))

    train_iterator.reset()
    val_iterator.reset()

    pruning_extension = ChainerPruningExtension(
        trial, "validation/main/loss", (1, "epoch")
    )

    result_info = train(
        model,
        train_iterator,
        val_iterator,
        3,
        device,
        lr,
        weight_decay,
        out_pref="models/neurocomp/hparam_search/",
        checkpoint_on=False,
        my_extensions=[pruning_extension],
        data_parallel=False,
    )

    best_val = result_info["best_val"]

    return best_val


def create_objective(gate_mode, train_iterator, val_iterator, device):
    return lambda x: objective(
        x, gate_mode, train_iterator, val_iterator, device
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gate_mode",
        type=str,
        choices=["none", "visual_gate", "language_gate", "multimodal_gate"],
    )
    parser.add_argument("--b_size", type=int, default=300)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    train_dataset, val_dataset = get_dataset(["train", "val"], subset_size=0.1)

    down_sampler = DownSampler(
        train_dataset.indices_positive, train_dataset.indices_negative
    )

    b_size = args.b_size

    train_iterator = SerialIterator(
        train_dataset, b_size, shuffle=None, order_sampler=down_sampler
    )

    val_iterator = SerialIterator(val_dataset, 2 * b_size, repeat=False)

    neptune.init(project_qualified_name="mayu-ot/VGP")
    neptune_exp = neptune.create_experiment(
        name=f"hparam_saerch_{args.gate_mode}",
        properties={
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "wd": os.getcwd(),
            "cmd": " ".join(sys.argv),
        },
    )

    monitor = opt_utils.NeptuneMonitor(neptune_exp)

    study = optuna.create_study(direction="maximize")

    obj_f = create_objective(
        args.gate_mode, train_iterator, val_iterator, args.device
    )

    study.optimize(obj_f, n_trials=100, callbacks=[monitor])

    opt_utils.log_study(study)


if __name__ == "__main__":
    main()
