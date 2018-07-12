# Multi-View GCN (MVGCN) for Brain Networks

## Overview
<p align="center"><img src="images/GCN.png" alt="Multi-View GCN" width="600"></p>

This repository contains TensorFlow code for implementing Multi-View Graph Convolutional Network for brain networks (DTI data). To do this, we train models using graph convolutional networks (GCNs) in multiple views to learn view-based feature representations. Then, view pooling is conducted for the purpose of multi-view feature fusion. The code is documented and should be easy to modify for your own applications.      

## Model
The objective function is established for a binary classification problem, which is matching vs. non-matching classes. Brain networks in the same group (Parkinson's Disease or Healthy Control) are labeled as matching pairs while brain networks from different groups are labeled as non-matching pairs. Hence, pairwise training samples are feed into the neural network. The figure depicts the pairwise learning architecture.  

<p align="center"><img src="images/overall.png" alt="Pairwise Learning Architecture" width="500"></p>

* The components utilized in the MVGCN are:
    * GCNs: incorporating coordinates of ROI (Region of Interest) with brain networks (DTI) together to learn interpretable representations.  
    * View pooling: combining view-based representations of DTI data that are obtained from various tractography algorithms.  
    * Pairwise matching: computing similarities by euclidean distance for sample pairs in terms of the pooled features.  
    * Softmax: classifying the output of pairwise matching layer into the matching vs. non-matching classes.

## Requirements
This package has the following requirements:
* An NVIDIA GPU.
* `Python3.x`
* [TensorFlow](https://github.com/tensorflow/tensorflow)

## Usage
To run MVGCN on your data, you need to: change the function of loading data in utils.py; define the names of multiple views in mvgcn.sh; set hyperparameters for MVGCN in mvgcn.sh; run the shell script mvgcn.sh
```bash
bash mvgcn.sh
```

## References: 
If you happen to use our work, please consider citing our paper:
```
@article{zhang2018multi,
  title={Multi-View Graph Convolutional Network and Its Applications on Neuroimage Analysis for Parkinson's Disease},
  author={Zhang, Xi and He, Lifang and Chen, Kun and Luo, Yuan and Zhou, Jiayu and Wang, Fei},
  journal={arXiv preprint arXiv:1805.08801},
  year={2018}
}
```
This paper can be accessed on : [Multi-View GCN] (https://arxiv.org/pdf/1805.08801.pdf)
