# Understanding Self-Supervised Pretraining Methods for Label-Scarce Ankle Fracture Classification

This repository contains all code for the master thesis: Understanding Self-Supervised Pretraining Methods for Label-Scarce Ankle Fracture Classification. 


The directory "multi_head_bone_classifier" contains all the code for the LuXry indexing classifier. 

The directory "SSL_pretraining" contains the implementation of the self-supervised learning methods:
- SparK
- MoCo v2
- MoCo v3
- VICReg(L)
- DINO v1

These methods have been implemented with the open-source code provided by the authors of each method:
- https://github.com/keyu-tian/SparK
- https://github.com/facebookresearch/moco
- https://github.com/facebookresearch/moco-v3
- https://github.com/facebookresearch/VICRegL
- https://github.com/facebookresearch/dino


The directory "ankle_fracture_classification" contains all code for the downstream [ankle fracture classification task](https://ieeexplore.ieee.org/abstract/document/9718182). Code related to data loading, model definition and training, etc. is an adaptation of: https://github.com/zhibinliao89/fracture_attention_guidance


