import numpy as np
import chainer
from .models.faster_rcnn import FasterRCNNExtractor
import progressbar
import json
from imageio import imread
from chainer.dataset.convert import to_device


def extract_frcnn_feat(df, device):
    model = FasterRCNNExtractor(n_fg_class=20, pretrained_model="voc07")
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    feat = np.zeros((len(df), 4096)).astype("f")

    bbox_iter = get_imagebbox_generator(df)
    N = df.image.unique().size

    j = 0

    indices = {}

    with progressbar.ProgressBar(max_value=N) as bar:

        for i, items in enumerate(bbox_iter):
            img, bbox, phrase, name = items

            # preprocess image
            img = img.transpose(2, 0, 1)
            img = model.prepare(img.astype(np.float32))
            img = to_device(device, img)

            bbox = to_device(device, bbox.astype(np.float32))
            bbox_indices = model.xp.zeros((len(bbox),)).astype("i")

            with chainer.no_backprop_mode(), chainer.using_config("train", False):
                y = model.extract(img[None, :], bbox, bbox_indices)

            y.to_cpu()
            n = len(y)
            feat[j : j + n, :] = y.data[:]
            indices[str(name)] = {k: v for k, v in zip(phrase, range(j, j + n))}
            j += n

            bar.update(i)

    return feat, indices


def get_alignment(bbox_data):
    align = {}

    for i in range(len(bbox_data)):
        phr = bbox_data.get_phrase(i)
        image = bbox_data.get_image_id(i)
        align.setdefault(str(image), {}).update({phr: i})

    return align


def get_imagebbox_generator(df):
    grouped = df.groupby(["image"])
    for name, group in grouped:
        im = imread("data/flickr30k-images/%s.jpg" % name)
        bbox = group[["ymin", "xmin", "ymax", "xmax"]].values
        phrase = group.phrase
        yield im, bbox, phrase, name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", "-s", type=str, default="val")
    parser.add_argument("--device", "-d", type=int, default=0)
    parser.add_argument("--method", type=str)
    args = parser.parse_args()

    if args.method == "ddpn":
        bbox_data = DDPNBBoxDataset(args.split)
    elif args.method == "plclc":
        bbox_data = PLCLCBBoxDataset(args.split)
    elif args.method == "gtroi":
        bbox_data = BBoxDataset(args.split)
    else:
        raise RuntimeError("invalid method name: %s" % args.method)

    df = bbox_data.df.copy()

    feat, align = extract_frcnn_feat(df, args.device)
    np.save(
        "data/phrase_localization/region_feat/%s_roi-frcnn/%s"
        % (args.method, args.split),
        feat,
    )

    json.dump(
        align,
        open(
            "data/phrase_localization/%s/vis_indices_%s.json"
            % (args.method, args.split),
            "w",
        ),
    )
