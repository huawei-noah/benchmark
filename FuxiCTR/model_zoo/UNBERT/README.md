# UNBERT

The official pytorch implementation for our paper [UNBERT: User-News Matching BERT for News Recommendation](https://www.ijcai.org/proceedings/2021/0462.pdf)

## Requirements
```bash
pip install -r requirements.txt
```

## Data preparation

For the MIND dataset, please download at https://msnews.github.io

File Name | Description
------------- | -------------
pretrainedModel  | pretrained model from huggingface, including config.json, pytorch_model.bin and vocab.txt
small/train  | MIND-small train dataset, including behaviors.tsv and news.tsv
small/dev  | MIND-small dev dataset
large/train  | MIND-large train dataset 
large/dev  | MIND-large dev dataset
large/test  | MIND-large test dataset

## Usage

```python
python run.py --mode train --split small --root ./data/ --pretrain data/bert-base-uncased/
```
For more parameter settings, please refer to `run.py`.
