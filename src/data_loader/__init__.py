from yacs.config import CfgNode
from typing import Callable, Iterable, List, Tuple
import chainer
from chainer.dataset import DatasetMixin
from src.data_loader.multimodal_dataloader import (
    PhraseOnlyDataLoader,
    MultiModalDataLoader,
    MultiModalIoUDataLoader,
    DownSampler,
)

# from src.data_loader.phrase_only_dataloader import PhraseOnlyDataLoader
from chainer.iterators import MultiprocessIterator
import numpy as np


def _phrase_cvrt(x):
    return np.asarray(x, np.int32)


def _do_nothing(x):
    return x


def get_dataset(
    name: str, split: str, subset_size: float = 1.0, use_iou: bool = False
) -> DatasetMixin:

    if name == "phrase-only":
        dataset_file = f"data/processed/ddpn/{split}.csv"
        dataset = PhraseOnlyDataLoader(dataset_file, split, subset_size)
    elif name == "multimodal":
        feat_type = "ddpn"

        dataset_file = f"data/processed/{feat_type}/{split}.csv"

        if use_iou:
            dataset = MultiModalIoUDataLoader(
                dataset_file, split, subset_size, feat_type
            )
        else:
            dataset = MultiModalDataLoader(
                dataset_file, split, subset_size, feat_type
            )

    return dataset


def build_single_dataloader(
    dataset: DatasetMixin, batch_size: int, downsample_on: bool, repeat: bool
) -> MultiprocessIterator:

    if downsample_on:
        down_sampler = DownSampler(
            dataset.indices_positive, dataset.indices_negative
        )
    else:
        down_sampler = None

    data_iterator = MultiprocessIterator(
        dataset,
        batch_size,
        order_sampler=down_sampler,
        repeat=repeat,
        shuffle=False,
    )
    return data_iterator


def build_parallel_dataloader(
    dataset: DatasetMixin,
    n: int,
    batch_size: int,
    downsample_on: bool,
    repeat: bool,
) -> List[MultiprocessIterator]:

    sub_datasets = chainer.datasets.split_dataset_n(dataset, n)

    iterators = []
    downsampler = None
    for sub_dataset in sub_datasets:
        if downsample_on:
            start = sub_dataset._start
            finish = sub_dataset._finish

            def align_indices(indices: List[int]) -> List[int]:
                return [x - start for x in indices if start <= x < finish]

            sub_indices_positive = align_indices(dataset.indices_positive)
            sub_indices_negative = align_indices(dataset.indices_negative)

            downsampler = DownSampler(
                sub_indices_positive, sub_indices_negative
            )

        data_iterator = MultiprocessIterator(
            sub_dataset,
            batch_size,
            order_sampler=downsampler,
            repeat=repeat,
            shuffle=None,
        )
        iterators.append(data_iterator)
    return iterators


def build_dataloader(split: str, cfg: CfgNode) -> List[MultiprocessIterator]:

    dataset = get_dataset(
        name=cfg.DATASET.NAME,
        split=split,
        subset_size=cfg.DATASET.TRAIN_SIZE,
        use_iou=cfg.MODEL.USE_IOU,
    )

    downsample_on = True if split == "train" else False
    repeat = True if split == "train" else False

    if cfg.TRAIN.N_PARALLEL > 1:
        iterators = build_parallel_dataloader(
            dataset,
            n=cfg.TRAIN.N_PARALLEL,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            downsample_on=downsample_on,
            repeat=repeat,
        )
    else:
        iterators = [
            build_single_dataloader(
                dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                downsample_on=downsample_on,
                repeat=repeat,
            )
        ]

    return iterators


def process_batch(
    batch: Iterable,
    preproc_f: List[Callable],
    postproc_f: List[Callable],
    device: chainer.backend.Device,
):

    assert len(batch[0]) == len(preproc_f)

    N = len(preproc_f)

    inputs = [[] for _ in range(N)]

    for b in batch:
        for i in range(N):
            x = preproc_f[i](b[i])
            inputs[i].append(x)

    for i in range(len(inputs)):
        inputs[i] = postproc_f[i](inputs[i])
        inputs[i] = device.send(inputs[i])

    return tuple(inputs)


@chainer.dataset.converter()
def multimodal_converter(
    batch: List[Tuple[List[int], List[int], np.ndarray, np.ndarray, int]],
    device: chainer.backend.Device,
):
    preproc_f = [
        _phrase_cvrt,
        _phrase_cvrt,
        _do_nothing,
        _do_nothing,
        _do_nothing,
    ]
    postproc_f = [
        _do_nothing,
        _do_nothing,
        lambda x: np.vstack(x),
        lambda x: np.vstack(x),
        lambda x: np.asarray(x, np.int32),
    ]
    return process_batch(batch, preproc_f, postproc_f, device)


@chainer.dataset.converter()
def multimodal_iou_converter(
    batch: List[
        Tuple[List[int], List[int], np.ndarray, np.ndarray, float, int]
    ],
    device: chainer.backend.Device,
):

    preproc_f = [
        _phrase_cvrt,
        _phrase_cvrt,
        _do_nothing,
        _do_nothing,
        _do_nothing,
        _do_nothing,
    ]

    postproc_f = [
        _do_nothing,
        _do_nothing,
        lambda x: np.vstack(x),
        lambda x: np.vstack(x),
        lambda x: np.asarray(x, np.float32)[:, None],
        lambda x: np.asarray(x, np.int32),
    ]
    return process_batch(batch, preproc_f, postproc_f, device)


@chainer.dataset.converter()
def phrase_converter(
    batch: List[Tuple[List[int], List[int], int]],
    device: chainer.backend.Device,
):
    preproc_f = [_phrase_cvrt, _phrase_cvrt, _do_nothing]

    postproc_f = [_do_nothing, _do_nothing, lambda x: np.asarray(x, np.int32)]

    return process_batch(batch, preproc_f, postproc_f, device)


def get_converter(data_name: str, use_iou: bool = False) -> Callable:
    if data_name == "multimodal":
        if use_iou:
            cnv_f = multimodal_iou_converter
        else:
            cnv_f = multimodal_converter
    elif data_name == "phrase-only":
        cnv_f = phrase_converter
    return cnv_f
