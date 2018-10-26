Chainer implementation of Visually Grounded Paraphrase (VGP) identification using phrase localization techniques.

# Data
Downlaod the Flickr30K dataset and copy `flickr30k-images/` under `data/` directory.

# Training Model
For training a model using the DDPN phrase locaization method, run

```
python script/training/train_pre_comp_feat.py -mt <model_type>
```
Here are the list of available *model_types*.

- lng: phrase-only model
- vis+gtroi: visual-only model using ground truth phrase localization results
- vis+lng+gtroi: ours using ground truth phrase localization results
- vis+plclcroi
- vis+lng+plclcroi
- vis+ddpnroi
- vis+lng+ddpnroi

Output files are created under `checkpoint/<date>/` with default settings.
Use `--out_pref` option to specify an output directory manually.

Example

```
python script/training/train_pre_comp_feat.py -mt vis+lng+ddpnroi --out_pref ddpnroi_result/
```

Output files will be wrote under `ddpnroi_result/<date>/`.

# Evaluation

```
python script/training/train_pre_comp_feat.py --eval <output_dir_name>
```