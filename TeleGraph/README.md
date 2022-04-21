# TeleGraph: A Benchmark Dataset for Hierarchical Link Prediction.
The repository contains the code and dataset for paper "TeleGraph: A Benchmark Dataset for Hierarchical Link Prediction" accepted by WebConf GLB 2022.

## Requirements:
* torch
* numpy
* torch_geometric
* sklearn
* scipy

## Data

 TeleGraph.gpickle is an attributed telecom network as illustructed ![plot] (https://github.com/huawei-noah/benchmark/blob/main/TeleGraph/alarmGraph.pdf)


## Runs:
 ### Run GAE models:
 ```bash
 cd gaes
 python gae.py --dataset Telecom --encoder GCN -epochs 4001 --lr 0.0001 --val_ratio 0.05 --test_ratio 0.10 --patience 200
 ```

### Run heuristics methods:
```bash
cd heuristics
python heuristics.py --dataset Telecom --batch_size 32 --use_heuristic CN
```


### Run SEAL:
```bash
cd SEAL
python seal.py --dataset Telecom --embedding DRNL --epoches 4001 --lr 0.0001 --weight_deccay 5e-4 --val_ratio 0.05 --test_ratio 0.10 --batch_size 32 --patience 20
```
