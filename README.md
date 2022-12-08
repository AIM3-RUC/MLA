# MLA

We release code and models of MLA<sub>CLIP</sub> on multilingual image-text retrieval. The models are trained on CC300K and finetuned on Multi30K. 

# Requirments
```
torch >= 1.7.1
transformers
opencv-python
```

## Pretrained models
The pretrained models (CLIP & M-BERT, for initialization) can be downloaded [here](https://drive.google.com/file/d/1lJU9RwuYTvEd9r9ReM9FyXRxgkxxTStx/view?usp=sharing)
```
unzip pretrained_model.zip
```

## Config & Checkpoint

Detail configuration files and checkpoints can be found [here](https://drive.google.com/file/d/1vUhJYAXrtzXtScG142zB6Id5GSt0uo0D/view?usp=sharing)
```
unzip expr.zip
```

## Data preparation

Download [annotations](https://drive.google.com/file/d/1LWp6RVAXUjHvljB0xUDgIg56jQRzPHcC/view?usp=sharing) and unzip it to `./dataset/`
```
unzip dataset.zip
```

**Conceptual Caption images** can be crawled [here](https://ai.google.com/research/ConceptualCaptions/download). After crawled from the web, place all images under `dataset/ConceptualCaption/images`

CC300K are used to train the released models. This subset can be found here `dataset/ConceptualCaption/cc300k.json`

**Flickr30K images** can be [requested here](https://forms.illinois.edu/sec/229675). Untar it to `dataset/Multi30k`
```
tar -xzvf flickr30k_images.tar.gz -C dataset/Multi30k
```
**MSCOCO images** can be downloaded and prepared with the following scripts:
```
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip

mkdir -p dataset/MSCOCO/images

unzip -d dataset/MSCOCO/images http://images.cocodataset.org/zips/train2014.zip 
unzip -d dataset/MSCOCO/images http://images.cocodataset.org/zips/val2014.zip 
unzip -d dataset/MSCOCO/images http://images.cocodataset.org/zips/test2014.zip 
```

## Train
```
# NLT stage
bash train.sh \
    expr/vitb32/NLT/config.json 0
# LE stage:
bash train.sh \
    expr/vitb32/LE/config.json 0
```
## Finetune on En (m30k)
```
bash train.sh \
    expr/vitb32/finetune-en-m30k/config.json 0
```
## Finetune on all (m30k)
```
bash train.sh \
    expr/vitb32/finetune-all-m30k/config.json 0
```
## Evaluate on zero-shot:
```
bash inference.sh \
    expr/vitb32/LE/pytorch_model.bin.1 \
    expr/vitb32/LE/pytorch_model.bin.1 \
    m30k+coco \
    expr/vitb32/LE/eval_m30k+coco
```
## Evaluate finetune-on-en
```
bash inference.sh \
    expr/vitb32/finetune-en-m30k/pytorch_model.bin.4 \
    expr/vitb32/LE/pytorch_model.bin.1 \
    m30k \
    expr/vitb32/finetune-en-m30k/eval_m30k
```

## Evaluate finetune-on-all
```
bash inference.sh \
    expr/vitb32/finetune-en-m30k/pytorch_model.bin.4 \
    expr/vitb32/finetune-all-m30k/pytorch_model.bin.10000 \
    m30k \
    expr/vitb32/finetune-all-m30k/eval_m30k
```