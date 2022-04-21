# TeleGraph: A Benchmark Dataset for Hierarchical Link Prediction.
The repository contains the code and dataset for paper "TeleGraph: A Benchmark Dataset for Hierarchical Link Prediction" accepted by WebConf GLB 2022. The paper is avaible at https://arxiv.org/abs/2204.07703

* Link prediction is a key problem for network-structured data, attracting considerable research efforts owing to its diverse applications. 
The current link prediction methods focus on general networks and are overly dependent on either the closed triangular structure
of networks or node attributes. Their performance on sparse or highly hierarchical networks has not been well studied.  On the other hand,   the available tree-like benchmark datasets are either simulated, with limited node information, or small in scale. To bridge this gap, we present a new benchmark dataset *TeleGraph* , a highly sparse and hierarchical telecommunication network associated with rich node attributes, for assessing and fostering the link inference techniques. 

## Requirements:
* torch
* numpy
* torch_geometric
* sklearn
* scipy

## Data

 * TeleGraph.gpickle is an attributed telecom network as illustructed in below:
<img src="https://github.com/huawei-noah/benchmark/blob/main/TeleGraph/alarmGraph.PNG" alt="telegraph"  align="middle"  width=40% height=40%>
 
 
 
  
 
 * unzip the TELECOM.zip to get telecom_graph.pt in which each node is associated with an 1x 240 feature vector regarding the alarm occoured or not. 


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
