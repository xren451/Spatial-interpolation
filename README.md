# AnchorGK 

Hi, Welcome to the official repository of the AnchorGK paper:  
**"Anchor-based Incremental and Stratified Graph Learning Framework for Inductive Spatio-Temporal Kriging".** 

## Comparison

**Please note that** [**OK**](https://link.springer.com/book/10.1007/978-3-662-05294-5), [**IDW**](https://www.sciencedirect.com/science/article/abs/pii/S0098300408000721) and [**GHM**](https://www.tandfonline.com/doi/full/10.1080/13658816.2022.2147530?scroll=top&needAccess=true) are statistical approaches.  
Among DL-based approaches, [**KCN**](https://arxiv.org/pdf/2306.09463), [**SSIN**](https://arxiv.org/pdf/2311.15530), [**INCREASE**](https://arxiv.org/abs/2302.02738), [**STGNP**](https://dl.acm.org/doi/pdf/10.1145/3580305.3599372) and [**KITS**](https://arxiv.org/pdf/2311.02565) exhibit limitations in effectively handling incomplete features across stations, accommodating spatially sparse observations, and operating in multivariate contexts.

The detailed comparisons can be found on:

### 🔍 Comparison of Spatial-Temporal Kriging and Graph-Based Models

| **Model** | **Strata**<br>**Aware** | **Inductive** | **Incremental** | **Sparse** | **Incomplete** | **Multi-**<br>**variate** | **Effi-**<br>**ciency** | **Strengths** | **Limitations** |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-- | :-- |
| [**OK**](https://link.springer.com/book/10.1007/978-3-662-05294-5)<br>(Ordinary Kriging) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Med | Classic baseline | Not scalable; ignores heterogeneity |
| [**IDW**](https://www.sciencedirect.com/science/article/abs/pii/S0098300408000721)<br>(Inverse Distance Weighting) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | High | Simple; no training | Ignores spatial correlation |
| [**GHM**](https://www.tandfonline.com/doi/full/10.1080/13658816.2022.2147530?scroll=top&needAccess=true)<br>(Generalised Heterogeneity Model) | ✔ | ✗ | ✗ | ✔ | ✗ | ✗ | Med | Stratified spatial modelling | Non-inductive; fixed graph |
| [**KCN**](https://arxiv.org/pdf/2306.09463) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Med | CNN-based spatial learning | Poor on missing & multi-variate data |
| [**IGNNK**](https://openreview.net/forum?id=jeBic1U1KXz) | ✗ | ✔ | ✗ | ✗ | ✗ | ✗ | High | Inductive kriging via GNN | No strata; no missing data support |
| [**INCREASE**](https://arxiv.org/abs/2302.02738) | ✗ | ✔ | ✗ | ✗ | ✗ | ✗ | Med | Good generalisation | No sparse or multivariate input support |
| [**STGNP**](https://dl.acm.org/doi/pdf/10.1145/3580305.3599372) | ✗ | ✔ | ✗ | ✗ | ✗ | ✔ | Med | Multivariate TS modelling | Poor handling of missing features |
| [**KITS**](https://arxiv.org/pdf/2311.02565) | ✗ | ✔ | ✔ | ✔ | ✗ | ✗ | Low | Handles sparsity; incremental | Biased by pseudo-nodes |
| [**SSIN**](https://arxiv.org/pdf/2311.15530) | ✗ | ✔ | ✗ | ✗ | ✗ | ✗ | Med | Lightweight spatial method | No support for missing or multivariate |
| **AnchorGK**<br>(Proposed) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | High | Full support for sparse,<br>incomplete, multivariate data | Scaling to larger graphs TBD |

## Introduction

**AnchorGK** is the advanced successor to [**KITS**](https://arxiv.org/pdf/2311.02565) and [**SSIN**](https://arxiv.org/pdf/2311.15530) in inductive spatio-temporal kriging. While these methods either depend on *dense node-wise correlations* or process features separately, they fail to effectively capture **region-level structures** and **cross-feature dependencies** in scenarios with **spatial sparsity** and **incomplete features**.  

To overcome these limitations, **AnchorGK** introduces an *incremental stratified spatial correlation component* to model **broad region-level spatial semantics** and a *dual-view graph learning layer* to integrate **cross-feature** and **cross-strata information**. This design allows the framework to exploit **diverse spatial** and **feature patterns**, thereby achieving more accurate and robust inference in **sparsely observed regions**.  

![arch.png](Figures/1.arch.png)
