neurocomp_vgp
==============================

## Install Dependencies

```
git clone https://github.com/ids-cv/neurocomp_vgp.git
cd neurocomp_vgp
docker build -t image_name_here .
```

## Data Preparation
```
wget https://visually-grounded-paraphrases.s3-ap-northeast-1.amazonaws.com/data/dictionary.txt -P data/processed/
wget https://visually-grounded-paraphrases.s3-ap-northeast-1.amazonaws.com/data/ddpn_data.zip -P data/processed/ddpn/
cd data/processed/ddpn/
unzip ddpn_data.zip
```

## Training

First, run container:

```
docker run --rm -it --gpus all \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -v /local/path/to/neurocomp_vgp:/app \
     /bin/bash
```

```
python src/experiment4/train.py data/configs/multimodal_gate.yaml
```

