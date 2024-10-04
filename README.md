# Master Thesis

## Installation

```bash
git clone https://github.com/stle1027/Masterarbeit.git
python3 -m venv virtual_environment
source virtual_environment/bin/activate
pip install -r Masterarbeit/requirements.txt
pip install -e ./Masterarbeit
```
## Demo

```bash
cd demo
python3 ./create_prototypes.py
(select "all")
cd ..
python3 ./demo/demo.py
```

For creating prototypes, SAM is required:
      Load the SAM model from: https://github.com/facebookresearch/segment-anything
If models are missing, download the ones provided by DE-ViT:
[Downloads.md](Downloads.md) for instructions to setup datasets and model checkpoints.

## Select part weights

```bash
cd demo
python3 ./select_part_prototype_weights.py
```

## Model Weakness Analysis
```bash
cd demo
python3 ./check_for_similar_prototypes.py
```

## Interactive tools/part segmentation
```bash
cd demo
python3 ./segmentTools.py
```

## Create gt file for own images
put test images in "demo/input/"
```bash
cd demo
python3 ./create_gt_with_sam.py
```

## Create gt file for insdet images
follow instructions in "demo/create_gt_for_insdet_dataset.ipynb"


## Training (DE-ViT)

```bash
vit=l task=ovd dataset=coco bash scripts/train.sh  # train open-vocabulary COCO with ViT-L

# task=ovd / fsod / osod
# dataset=coco / lvis
# vit=s / b / l

# few-shot env var `shot = 5 / 10 / 30`
vit=l task=fsod shot=10 bash scripts/train.sh 

# one-shot env var `split = 1 / 2 / 3 / 4`
vit=l task=osod split=1 bash script/train.sh

# detectron2 options can be provided through args, e.g.,
task=ovd dataset=lvis bash scripts/train.sh MODEL.MASK_ON True # train lvis with mask head

# another env var is `num_gpus = 1 / 2 ...`, used to control
# how many gpus are used
```

## RPN Training (COCO) (DE-ViT)

```bash
bash scripts/train_rpn.sh  ARG
# change ARG to ovd / os1 / os2 / os3 / os4 / fs14
# corresponds to open-vocabulary / one-shot splits 1-4 / few-shot
```

Check [Tools.md](Tools.md) for intructions to build prototype and prepare weights.

## Acknowledgement
This master's thesis is based on the code of DE-ViT:

[Arxiv Paper](https://arxiv.org/abs/2309.12969)
```
@misc{zhang2023detect,
      title={Detect Every Thing with Few Examples}, 
      author={Xinyu Zhang and Yuting Wang and Abdeslam Boularias},
      year={2023},
      eprint={2309.12969},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



