<h1 align="center">Discrete latent variable models estimation</h1>
<p align="center"> <span style="font-size: 14px;"><em><strong>Luca Brusa &middot; Fulvia Pennoni &middot; Francesco Bartolucci</strong></em></span> </p>
<br>

The `estDLVM` package provides functions to perform maximum likelihood and approximate maximum likelihood estimation in the context of discrete latent variable models. For the latent class and hidden Markov models three versions of the expectation-maximization algorithm are implemented: standard (EM), tempered (TEM), and evolutionary (EEM). For the stochastic block model two versions of the variational expectation-maximization algorithm are implemented: standard (VEM) and evolutionary (EVEM).

To install the `estDLVM` package directly from GitHub:
```r
# install.packages("devtools")
require(devtools)
devtools::install_github("LB1304/estDLVM")
```

To download the .tar.gz file (for manual installation) use [this link](https://github.com/LB1304/estDLVM/archive/main.tar.gz).
