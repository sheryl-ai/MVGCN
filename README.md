# Multi-View GCN (MVGCN) for Brain Networks

## Overview
<p align="center"><img src="images/GCN.png" alt="Multi-View GCN" width="300"></p>

This repository contains TensorFlow code for implementing Multi-View Graph Convolutional Network for brain networks (DTI data). To do this, we train models using graph convolutional networks (GCNs) in multiple views to learn view-based feature representations. Then, view pooling is conducted for the purpose of multi-view feature fusion. The code is documented and should be easy to modify for your own applications.      

## Model
The objective function is established for a binary classification problem, which is matching vs. non-matching classes. brain networks in the same group (PD or HC) are labeled as matching pairs while brain networks from different groups are labeled as non-matching pairs. Hence, pairwise training samples are feed into the neural network. The figure depicts the pairwise learning architecture.  

<p align="center"><img src="images/overall.png" alt="Pairwise Learning Architecture" width="300"></p>

* The components utilized in the MVGCN are:
    * GCNs: incorporating coordinates of ROI (Region of Interest) with brain networks (DTI) together to learn interpretable representations.  
    * View pooling: combining view-based representations of DTI data obtained from various tractography algorithms.  
    * Pairwise matching: computing similarities by euclidean distance for sample pairs in terms of the pooled features.  
    * Softmax: classifying the output of pairwise matching layer into the matching vs. non-matching classes.

## Requirements
    This package has the following requirements:
    * `python3.x`
    * An NVIDIA GPU w/.
    * [TensorFlow](https://github.com/tensorflow/tensorflow)
        * Used for automatic differentiation

### Usage

To run MVGCN on your data, you need to: change the function of loading data in utils.py; define the names of multiple views in mvgcn.sh; set hyperparameters for MVGCN in mvgcn.sh; run the shell script mvgcn.sh
```bash
bash mvgcn.sh
```
