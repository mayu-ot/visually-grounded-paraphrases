from typing import List, Sequence, Dict
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from multiprocessing import Pool

VGENOME_ROOT = "/home/mayu-ot/Data/VisualGenome"


def _iou(x1: Sequence[int], x2: Sequence[int]) -> float:
    """
    x1, x2 = x_min, y_min, x_max, y_max
    """
    tl = np.maximum(x1[:2], x2[:2])
    br = np.minimum(x1[2:], x2[2:])
    area_i = np.prod(br - tl) * (tl < br).all()
    area_a = np.prod(x1[2:] - x1[:2])
    area_b = np.prod(x2[2:] - x2[:2])

    return area_i / (area_a + area_b - area_i)


def get_paraphrase_pair(x: Dict) -> Dict:
    im_id = x["id"]
    regions = x["regions"]
    img_size = x["size"]

    bboxes = []
    for r in regions:
        x, y, w, h = r["x"], r["y"], r["width"], r["height"]
        bboxes.append((x, y, x + w, y + h))

    item = {"image": im_id, "VGPs": [], "non-VGPs": []}

    # compute pairwise IoU
    iou_mat = squareform(pdist(bboxes, metric=_iou))

    # get VGPs
    thresh = 0.9
    idx1, idx2 = np.where(iou_mat > thresh)

    for i1, i2 in zip(idx1, idx2):
        if i1 < i2:

            # skip pairs of the same phrase
            if regions[i1]["phrase"].lower() == regions[i2]["phrase"].lower():
                continue

            # skip too large bounding box
            size_a = regions[i1]["width"] * regions[i1]["height"]
            size_b = regions[i2]["width"] * regions[i2]["height"]
            if max(size_a, size_b) / img_size > 0.6:
                continue

            item["VGPs"].append(
                {
                    "phrase_a": regions[i1]["phrase"],
                    "phrase_b": regions[i2]["phrase"],
                    "bbox_a": bboxes[i1],
                    "bbox_b": bboxes[i2],
                }
            )

    # get non-VGPs (downsample to the # of VGPS)
    thresh = 0.5
    idx1, idx2 = np.where(iou_mat < thresh)
    sample_idx = np.random.permutation(len(idx1))[: len(item["VGPs"])]
    idx1 = idx1[sample_idx]
    idx2 = idx2[sample_idx]

    for i1, i2 in zip(idx1, idx2):
        if i1 < i2:
            item["non-VGPs"].append(
                {
                    "phrase_a": regions[i1]["phrase"],
                    "phrase_b": regions[i2]["phrase"],
                    "bbox_a": bboxes[i1],
                    "bbox_b": bboxes[i2],
                }
            )

    return item


def construct_dataset() -> List[Dict]:
    """
    get VGP and non-VGP from Visual Genome
    """

    data = json.load(open(f"{VGENOME_ROOT}/region_descriptions.json"))
    meta_data = json.load(open(f"{VGENOME_ROOT}/image_data.json"))

    for i in range(len(data)):
        size = meta_data[i]["height"] * meta_data[i]["width"]
        data[i]["size"] = size

    with Pool(8) as p:
        imap = p.imap(get_paraphrase_pair, data, 10)
        dataset = list(tqdm(imap, total=len(data)))

    return dataset


def main():
    dataset = construct_dataset()
    json.dump(dataset, open("data/interim/visual_genome_VGPs.json", "w"))


if __name__ == "__main__":
    main()
