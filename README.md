<h1 align="center">Discrete latent variable models estimation</h1>
<p align="center"> <span style="font-size: 14px;"><em><strong>Luca Brusa</strong></em></span> </p>
<br>

The `estDLVM` package provides functions to perform maximum likelihood estimation in the field of discrete latent variable models. Three different variants of the Expectation-Maximization algorithm are implemented (standard, tempered, and evolutionary), and many models are taken into account (Latent Class, Hidden Markov, and Stochastic Block models).

To install the `estDLVM` package directly from GitHub:
```r
# install.packages("devtools")
require(devtools)
devtools::install_github("LB1304/estDLVM", ref = "main", 
                         auth_token = "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")
```
