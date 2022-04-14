# TeleGraph: A Benchmark Dataset for Hierarchical Link Prediction.
The repository contains the code and dataset for paper "TeleGraph: A Benchmark Dataset for Hierarchical Link Prediction" accepted by WebConf GLB 2022.

## Requirements:
* torch
* numpy
* torch_geometric
* sklearn
* scipy

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

<!-- ### Run LGLP:
```bash
cd LGLP/LGLP
python Main.py --data_name Telecom --max-train-num 10000 --hop 1 --max-nodes-per-hop 100
```

### Run NeoGNNs:
```bash
cd NeoGnn
python neognn.py --dataset Telecom --num_layers 2 --hidden_channels 256 --batch_size 256 --test_batch_size 256 --lr 0.01 --epochs 200 --runs 10
``` -->

### Run SEAL:
```bash
cd SEAL
python seal.py --dataset Telecom --embedding DRNL --epoches 101 --lr 0.0001 --weight_deccay 5e-4 --val_ratio 0.05 --test_ratio 0.10 --batch_size 32 --patience 20
```
