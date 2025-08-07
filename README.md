# AnchorGK 

Hi, Welcome to the official repository of the AnchorGK paper: "Anchor-based Incremental and Stratified Graph Learning Framework for Inductive Spatio-Temporal Kriging". 

## Comparison

**Please note that** [**OK**](https://link.springer.com/book/10.1007/978-3-662-05294-5), [**IDW**](https://www.sciencedirect.com/science/article/abs/pii/S0098300408000721) and [**GHM**](https://www.tandfonline.com/doi/full/10.1080/13658816.2022.2147530?scroll=top&needAccess=true) are statistical approaches.
Among DL-based approaches, [**KCN**](https://arxiv.org/pdf/2306.09463), [**SSIN**](https://arxiv.org/pdf/2311.15530), [**INCREASE**](https://arxiv.org/abs/2302.02738), [**STGNP**](https://dl.acm.org/doi/pdf/10.1145/3580305.3599372) and [**KITS**](https://arxiv.org/pdf/2311.02565) exhibit limitations in effectively handling incomplete features across stations, accommodating spatially sparse observations, and operating in multivariate contexts.

The detailed comparisons can be found on:

### 🔍 **Comparison of Spatial-Temporal Kriging and Graph-Based Models**

| **Model** | **Strata**<br>**Awareness** | **Inductive**<br>**Learning** | **Training**<br>**Strategy** | **Spatial**<br>**Sparsity** | **Incomplete**<br>**Features** | **Multivariate**<br>**Support** | **Efficiency** | **Strengths** | **Limitations** |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-- | :-- |
| **OK**<br>(Ordinary Kriging) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Moderate | Classic geostatistical<br>baseline | Not scalable;<br>ignores heterogeneity |
| **IDW**<br>(Inverse Distance Weighting) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | High | Simple; no training<br>required | Ignores spatial<br>correlation |
| **GHM**<br>(Generalised Heterogeneity Model) | ✔ | ✗ | ✗ | ✔ | ✗ | ✗ | Medium | Captures stratified<br>spatial variation | Non-inductive;<br>fixed graph |
| [**KCN**](https://arxiv.org/pdf/2306.09463) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Medium | CNN-based spatial<br>feature learning | Fails on missing<br>or multivariate data |
| [**IGNNK**](https://openreview.net/forum?id=jeBic1U1KXz) | ✗ | ✔ | ✗ | ✗ | ✗ | ✗ | High | Inductive kriging<br>via GNN | No support for strata;<br>incomplete features |
| [**INCREASE**](https://arxiv.org/abs/2302.02738) | ✗ | ✔ | ✗ | ✗ | ✗ | ✗ | Medium | Inductive generalisation<br>ability | Lacks support for<br>sparse and multivariate |
| [**STGNP**](https://dl.acm.org/doi/pdf/10.1145/3580305.3599372) | ✗ | ✔ | ✗ | ✗ | ✗ | ✔ | Medium | Multivariate<br>temporal modeling | Poor handling of<br>incomplete features |
| [**KITS**](https://arxiv.org/pdf/2311.02565) | ✗ | ✔ | ✔ | ✔ | ✗ | ✗ | Low | Supports sparsity<br>and incremental updates | Biased by pseudo<br>nodes; no multivariate |
| [**SSIN**](https://arxiv.org/pdf/2311.15530) | ✗ | ✔ | ✗ | ✗ | ✗ | ✗ | Medium | Lightweight and<br>spatially aware | Lacks support for<br>missing/multivariate |
| **AnchorGK**<br>(Proposed) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | High | Full support for<br>sparse, incomplete,<br>multivariate data | Scalability to<br>larger graphs TBD |


