# TeleGraph: A Benchmark Dataset for Hierarchical Link Prediction.
The repository contains the code and dataset for paper "TeleGraph: A Benchmark Dataset for Hierarchical Link Prediction" accepted by WebConf GLB 2022. The paper is avaible at https://arxiv.org/abs/2204.07703

* Link prediction is the problem of detecting the presence of a connection between two entities in a network. Research fields, ranging from network science to machine learning and data mining, have taken a great interest in link prediction task. Given that hierarchical patterns found in many real-world applications while the available research datasets are inadequate, in this work, we present a new real-world dataset *TeleGraph*, which is a medium sized and heterogeneous telecommunication network with a rich set of attributes.   Our descriptive analysishas demonstrated it is highly hierarchical and sparse, which makes the heuristic measures fail to work. We verified this precognition by a series of experiments. Our findings show that most of the available algorithms fail to produce the satisfactory performance on this tree-like dataset except the subgraph-based GNN-models. More specifically, the results of a series heuristic measures are even close to random guesses, which calls for special attention in practice.  We believe *TeleGraph* can serve as an useful benchmark to assess and foster novel link prediction and node embedding techniques. 

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
