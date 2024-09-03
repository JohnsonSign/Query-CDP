# Query-CP

Learnable Query Contrast and Spatio-temporal Prediction on Point Cloud Video Pre-training

# Introduction
In this paper, we propose a pre-training framework Query-CP to learn the representations of point cloud videos through multiple self-supervised pretext tasks. First, token-level contrast is 
developed to predict future features under the guidance of historical information. Using a position-guided autoregressor with learnable queries, the predictions are directly contrasted with 
corresponding targets in the high-level feature space to capture fine-grained semantics. Second, performing only contrastive learning fails to fully explore the complementary structures and 
dynamics information. To alleviate this, a decoupled spatio-temporal prediction task is designed, where we use a spatial branch to predict low-level features and a temporal branch to predict 
timestamps of the target sequence explicitly.

# Installation
The code is tested with Python 3.7.12, PyTorch 1.7.1, GCC 9.4.0, and CUDA 10.2. Compile the CUDA layers for PointNet++: https://arxiv.org/abs/1706.02413

cd modules

python setup.py install

# The steps for performing the experiments

(1) pretraining: python 0-pretrain-msr.py


(2) Finetuning: python main-msr.py

# Datasets

MSRAction 3D 

NTU RGB-D

Synthia 4D

Please refer to the related repositories for downloading these datasets.

# Related Repositories

We thank the authors of related repositories:

PSTNet: https://github.com/hehefan/Point-Spatio-Temporal-Convolution

P4Transformer: https://github.com/hehefan/P4Transformer
