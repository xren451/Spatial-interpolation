# AnchorGK 

Hi, Welcome to the official repository of the AnchorGK paper: "Anchor-based Incremental and Stratified Graph Learning Framework for Inductive Spatio-Temporal Kriging". 

##Comparison

**Please note that** [**OK**](https://link.springer.com/book/10.1007/978-3-662-05294-5), [**IDW**](https://www.sciencedirect.com/science/article/abs/pii/S0098300408000721) and [**GHM**](https://www.tandfonline.com/doi/full/10.1080/13658816.2022.2147530?scroll=top&needAccess=true) are statistical approaches.
Among DL-based approaches, [**KCN**](https://arxiv.org/pdf/2306.09463), [**SSIN**](https://arxiv.org/pdf/2311.15530), [**INCREASE**](https://arxiv.org/abs/2302.02738), [**STGNP**](https://dl.acm.org/doi/pdf/10.1145/3580305.3599372) and [**KITS**](https://arxiv.org/pdf/2311.02565) exhibit limitations in effectively handling incomplete features across stations, accommodating spatially sparse observations, and operating in multivariate contexts.

The detailed comparisons can be found on:

### 🔍 **Comparison of Spatial-Temporal Kriging and Graph-Based Models**

| **Model** | **Strata Awareness** | **Inductive Learning** | **Training Strategy** | **Spatial Sparsity** | **Incomplete Features** | **Multivariate** | **Efficiency** | **Strengths** | **Limitations** |
| :-- | :--------------------: | :----------------------: | :--------------------: | :-------------------: | :----------------------: | :---------------: | :------------: | :-----------: | :-------------- |
| **OK**<br>(Ordinary Kriging) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Moderate | Classic geostatistical baseline | Not scalable; ignores heterogeneity |
| **IDW**<br>(Inverse Distance Weighting) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | High | Simple, no training required | Ignores spatial correlation; no uncertainty |
| **GHM**<br>(Generalised Heterogeneity Model) | ✔ | ✗ | ✗ | ✔ | ✗ | ✗ | Medium | Captures stratified spatial variation | Non-inductive; fixed spatial graph |
| [**KCN**](https://arxiv.org/pdf/2306.09463) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Medium | CNN-based spatial learning | Fails on missing data and multivariate input |
| [**IGNNK**](https://openreview.net/forum?id=jeBic1U1KXz) | ✗ | ✔ | ✗ | ✗ | ✗ | ✗ | High | Inductive GNN-based kriging | Ignores strata; lacks incomplete feature support |
| [**INCREASE**](https://arxiv.org/abs/2302.02738) | ✗ | ✔ | ✗ | ✗ | ✗ | ✗ | Medium | Strong inductive ability | Weak for sparse or multivariate settings |
| [**STGNP**](https://dl.acm.org/doi/pdf/10.1145/3580305.3599372) | ✗ | ✔ | ✗ | ✗ | ✗ | ✔ | Medium | Multivariate spatiotemporal learning | Incomplete feature handling is weak |
| [**KITS**](https://arxiv.org/pdf/2311.02565) | ✗ | ✔ | ✔ | ✔ | ✗ | ✗ | Low | Supports sparsity and incremental updates | Biased by pseudo nodes; lacks multivariate support |
| [**SSIN**](https://arxiv.org/pdf/2311.15530) | ✗ | ✔ | ✗ | ✗ | ✗ | ✗ | Medium | Lightweight and spatially aware | No support for incomplete or multivariate data |
| **AnchorGK**<br>(Proposed) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | High | Full support for spatial and feature heterogeneity | Scaling to ultra-large graphs remains to be explored |

