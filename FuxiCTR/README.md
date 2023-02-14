# FuxiCTR

Click-through rate (CTR) prediction is a critical task for many industrial applications such as online advertising, recommender systems, and sponsored search. FuxiCTR provides an open-source library for CTR prediction, with key features in configurability, tunability, and reproducibility. We hope this project could benefit both researchers and practitioners with the goal of open benchmarking for CTR prediction tasks.

If you find FuxiCTR useful in your research, please kindly cite the following papers: 

> Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He. [Open Benchmarking for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794). *The 30th ACM International Conference on Information and Knowledge Management (CIKM)*, 2021. [[Bibtex](https://dblp.org/rec/conf/cikm/ZhuLYZH21.html?view=bibtex)]

> Jieming Zhu, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Xi Xiao, Rui Zhang. [BARS: Towards Open Benchmarking for Recommender Systems](https://arxiv.org/abs/2205.09626). *The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)*, 2022. [[Bibtex](https://dblp.org/rec/conf/sigir/ZhuDSMLCXZ22.html?view=bibtex)]

## Model Zoo

| Publication        | Model                                    | Paper                                                                                                                                                                       | Available          |
|:------------------:|:----------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------------:|
| WWW'07             | [LR](./model_zoo/LR)                     | [Predicting Clicks: Estimating the Click-Through Rate for New Ads](https://dl.acm.org/citation.cfm?id=1242643)                                                              | :heavy_check_mark: |
| ICDM'10            | [FM](./model_zoo/FM)                     | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)                                                                                        | :heavy_check_mark: |
| CIKM'15            | [CCPM](./model_zoo/CCPM)                 | [A Convolutional Click Prediction Model](http://www.escience.cn/system/download/73676)                                                                                      | :heavy_check_mark: |
| RecSys'16          | [FFM](./model_zoo/FFM)                   | [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/citation.cfm?id=2959134)                                                                         | :heavy_check_mark: |
| RecSys'16          | [YoutubeDNN](./model_zoo/DNN)            | [Deep Neural Networks for YouTube Recommendations](http://art.yale.edu/file_columns/0001/1132/covington.pdf)                                                                | :heavy_check_mark: |
| DLRS'16            | [Wide&Deep](./model_zoo/WideDeep)        | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                                        | :heavy_check_mark: |
| ICDM'16            | [IPNN](./model_zoo/PNN)                  | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                                          | :heavy_check_mark: |
| KDD'16             | [DeepCrossing](./model_zoo/DeepCrossing) | [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)                             | :heavy_check_mark: |
| NIPS'16            | [HOFM](./model_zoo/HOFM)                 | [Higher-Order Factorization Machines](https://papers.nips.cc/paper/6144-higher-order-factorization-machines.pdf)                                                            | :heavy_check_mark: |
| IJCAI'17           | [DeepFM](./model_zoo/DeepFM)             | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)                                                                 | :heavy_check_mark: |
| SIGIR'17           | [NFM](./model_zoo/NFM)                   | [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777)                                                                 | :heavy_check_mark: |
| IJCAI'17           | [AFM](./model_zoo/AFM)                   | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/0435.pdf)                    | :heavy_check_mark: |
| ADKDD'17           | [DCN](./model_zoo/DCN)                   | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                                           | :heavy_check_mark: |
| WWW'18             | [FwFM](./model_zoo/FwFM)                 | [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)                                      | :heavy_check_mark: |
| KDD'18             | [xDeepFM](./model_zoo/xDeepFM)           | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                                               | :heavy_check_mark: |
| KDD'18             | [DIN](./model_zoo/DIN)                   | [Deep Interest Network for Click-Through Rate Prediction](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction)         | :heavy_check_mark: |
| CIKM'19            | [FiGNN](./model_zoo/FiGNN)               | [FiGNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction](https://arxiv.org/abs/1910.05552)                                                       | :heavy_check_mark: |
| CIKM'19            | [AutoInt/AutoInt+](./model_zoo/AutoInt)  | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                                                      | :heavy_check_mark: |
| RecSys'19          | [FiBiNET](./model_zoo/FiBiNET)           | [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433)                                | :heavy_check_mark: |
| WWW'19             | [FGCNN](./model_zoo/FGCNN)               | [Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/abs/1904.04447)                                                    | :heavy_check_mark: |
| AAAI'19            | [HFM/HFM+](./model_zoo/HFM)              | [Holographic Factorization Machines for Recommendation](https://ojs.aaai.org//index.php/AAAI/article/view/4448)                                                             | :heavy_check_mark: |
| AAAI'19            | [DIEN](./model_zoo/DIEN)                 | [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672)                                                                       | :heavy_check_mark: |
| DLP-KDD'19         | [BST](./model_zoo/BST)                   | [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874)                                                                  | :heavy_check_mark: |
| Neural Networks'20 | [ONN](./model_zoo/ONN)                   | [Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579)                                                                            | :heavy_check_mark: |
| AAAI'20            | [AFN/AFN+](./model_zoo/AFN)              | [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768)                                       | :heavy_check_mark: |
| AAAI'20            | [LorentzFM](./model_zoo/LorentzFM)       | [Learning Feature Interactions with Lorentzian Factorization](https://arxiv.org/abs/1911.09821)                                                                             | :heavy_check_mark: |
| WSDM'20            | [InterHAt](./model_zoo/InterHAt)         | [Interpretable Click-through Rate Prediction through Hierarchical Attention](https://dl.acm.org/doi/10.1145/3336191.3371785)                                                | :heavy_check_mark: |
| DLP-KDD'20         | [FLEN](./model_zoo/FLEN)                 | [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/abs/1911.04690)                                                                                      | :heavy_check_mark: |
| WWW'21             | [FmFM](./model_zoo/FmFM)                 | [FM^2: Field-matrixed Factorization Machines for Recommender Systems](https://arxiv.org/abs/2102.12994)                                                                     | :heavy_check_mark: |
| WWW'21             | [DCN-V2](./model_zoo/DCNv2)              | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)                                      | :heavy_check_mark: |
| CIKM'21            | [DESTINE](./model_zoo/DESTINE)           | [Disentangled Self-Attentive Neural Networks for Click-Through Rate Prediction](https://arxiv.org/abs/2101.03654)                                                           | :heavy_check_mark: |
| CIKM'21            | [EDCN](./model_zoo/EDCN)                 | [Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf) | :heavy_check_mark: |
| DLP-KDD'21         | [MaskNet](./model_zoo/MaskNet)           | [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/abs/2102.07619)                                          | :heavy_check_mark: |
| SIGIR'21           | [SAM](./model_zoo/SAM)                   | [Looking at CTR Prediction Again: Is Attention All You Need?](https://arxiv.org/abs/2105.05563)                                                                             | :heavy_check_mark: |
| KDD'21             | [AOANet](./model_zoo/AOANet)             | [Architecture and Operation Adaptive Network for Online Recommendations](https://dl.acm.org/doi/10.1145/3447548.3467133)                                                    | :heavy_check_mark: |
| IJCAI'21           | [UNBERT](./model_zoo/UNBERT)             | [UNBERT: User-News Matching BERT for News Recommendation](https://www.ijcai.org/proceedings/2021/0462.pdf)                                                                  | :heavy_check_mark: |
| CIKM'22            | [SDIM](./model_zoo/SDIM)                 | [Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction](https://arxiv.org/abs/2205.10249)                                                        | :heavy_check_mark: |

## Dependency

FuxiCTR has the following dependent requirements. 

+ pytorch 1.10+
+ python 3.6+
+ pyyaml 5.1+
+ scikit-learn
+ pandas
+ numpy
+ h5py
+ tqdm

## Get Started

One can easily run each model in the model zoo following the commands below, which is a demo for running DCN. In addition, users can modify the dataset config and model config files to run on their own datasets or on new hyper-parameters.

```
cd model_zoo/DCN/DCN_torch
python run_expid --expid DCN_test --gpu 0
```

## Join Us

We have open positions for internships and full-time jobs. If you are interested in research and practice in recommender systems, please send your CV to jamie.zhu@huawei.com.
