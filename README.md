# AnchorGK: Anchor Station-based Graph Learning Framework for Spatio-Temporal Kriging

This repository contains the code and datasets for the paper "AnchorGK: Anchor Station-based Graph Learning Framework for
Spatio-Temporal Kriging". In this paper, we propose a new method for spatio-temporal kriging considering the sparse spatial distribution of stations and the availability of features at different stations.

The overall architecture can be found on the following figure.

<p align="center">
  <img src="Fig/arch.png" alt="1 Architecture">
</p>

The main results on spatial kriging task can be found here:

<p align="center">
  <img src="Fig/results_AnchorGK.png" alt="2 Result">
</p>

Datasets:
UScoast: We download the UScoast dataset from [here](https://www.ndbc.noaa.gov/data/).
Shenzhen: The Shenzhen dataset is obtained from [here](https://github.com/xren451/DAMR/tree/main).

Baseline models:
Please run the baseline models.

Hyperparameter:
On geostatistical methods, we set the power of 2 for IDW following the same baseline setting in SSIN; We keep default setting on OKL and OKG. On SSIN, the number of attention heads $H$ and hidden states $d_{k}$ in attention scheme are set to $2$ and $16$. The dimension of feed-forward network $d_{f}$ is set to 256. In IGNNK, the hidden state in GNN is set to 25, the number of unknown station percentage and mask node during training are both set to $20\%$, the diffusion convolution step is set to $1$, the number of max training episode is set to 750. On KCN, the hidden size, kernel length and dropout rate are set to $20, 0.25$ and $1.05$, respectively.
On AnchorGK, the number of neighbors in each king graph is set to $5$; The decay coefficient is tuned by the range (0,1.6,0.1); The hidden dimension in MOE model is set to 16; The $\alpha$, $\beta$ and $\kappa$ are set to 0.1, 2.0 and 0.1, respectively in KFE. The hidden state in Graph Convolution Network is set to $16$. The kernel function is set to linear on local kriging. We update the parameters in Sub-region Spatial Correlation Component following Markov chain Monte Carlo (MCMC). All experiments were run on Intel(R) Core(TM) i9-13900HX CPU, 32GB RAM, and NVIDIA RTX 4080 GPU.
