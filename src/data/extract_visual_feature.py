import numpy as np
import chainer
from src.models.faster_rcnn import FasterRCNNExtractor
import pandas as pd
from imageio import imread
from chainer.dataset.convert import to_device
from tqdm import tqdm

chainer.config.cv_resize_backend = "cv2"

def extract_frcnn_feat(df, device):
    df = pd.read_csv(df)
    model = FasterRCNNExtractor(n_fg_class=20, pretrained_model="voc07")
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    feat = np.zeros((len(df), 4096)).astype("f")

    bbox_iter = get_imagebbox_generator(df)

    for items in tqdm(bbox_iter):
        img, bbox, indices = items

        # preprocess image
        img = img.transpose(2, 0, 1)
        img = model.prepare(img.astype(np.float32))
        img = to_device(device, img)

        bbox = to_device(device, bbox.astype(np.float32))
        bbox_indices = model.xp.zeros((len(bbox),)).astype("i")

        with chainer.no_backprop_mode(), chainer.using_config("train", False):
            y = model.extract(img[None, :], bbox, bbox_indices)

        y.to_cpu()
        feat[indices] = y.data[:]

    return feat

def get_imagebbox_generator(df):
    grouped = df.groupby(["image"])
    for name, group in grouped:
        im = imread(f"data/raw/flickr30k-images/{name}.jpg")
        bbox = group[["ymin", "xmin", "ymax", "xmax"]].values
        indices = group.index
        yield im, bbox, indices


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
