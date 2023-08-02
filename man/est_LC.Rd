\name{est_LC}
\alias{est_LC}
\title{Maximum likelihood estimation of the latent class model}
\description{
  This function performs maximum likelihood estimation of the latent class model through different versions of the expectation-maximization algorithm: standard (EM), tempered (TEM), and evolutionary (EEM).
}
\usage{
est_LC(data, k, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, sv = NULL, algorithm = c("EM", "TEM", "EEM"), 
       temperingOptions = list(profile = NULL, alpha = NULL, beta = NULL, rho = NULL, T0 = NULL),
       evolutionaryOptions = list(n_parents = NULL, n_children = NULL, prob_mut = NULL, R = NULL))
}
\arguments{
  \item{data}{Data, stored either as a \code{matrix} or as a \code{data.frame}}
  \item{k}{Number of latent components}
  \item{tol_lk, tol_theta}{Tolerance levels for convergence}
  \item{maxit}{Maximum number of iterations of the algorithm}
  \item{sv}{List of initial model parameters; default is \code{NULL} and starting values are randomly generated}
  \item{algorithm}{Available versions of the expectation-maximization algorithm: standard (\code{"EM"}), tempered (\code{"TEM"}), and evolutionary (\code{"EEM"})}
  \item{temperingOptions}{List of constants required for the tempering version of the alforithm}
  \item{evolutionaryOptions}{List of constants required for the evolutionary version of the algorithm}
}
\value{
  \item{LogLik}{Log-likelhood at convergence of the algorithm}
  \item{LogLik_vec}{Vector to trace the log-likelihood evolution across iterations}
  \item{it}{Number of iterations until convergence}
  \item{piv}{Estimated vector of weights of the latent classes}
  \item{Phi}{Array of the conditional response probabilities for every item and latent class}
  \item{k}{Number of latent components}
  \item{np}{Total number of estimated parameters}
  \item{V}{Matrix of the posterior probabilities for each response configuration and latent class}
  \item{aic}{Value of the Akaike Information Criterion for model selection}
  \item{bic}{Value of the Bayesian Information Criterion for model selection}
  \item{call}{Command used to call the function}
}
\examples{
  require(estDLVM)
  ## Import the co-authorship dataset
  HG <- import_Hypergraph(file_name = "HG_coauth.txt", method = "full")
  
  ## Estimation of the HSBM with 2 latent groups, random initialization and full model (not parallelized version)
  est <- HSBM(Hypergraph = HG, Q = 2, tol = 1e-6, maxit_VEM = 100, maxit_FP = 100)
  ## Estimation of the HSBM with 2 latent groups, random initialization and full model (parallelized version with 30 cores)
  est <- HSBM(Hypergraph = HG, Q = 2, tol = 1e-6, maxit_VEM = 100, maxit_FP = 100, n_threads = 30)
  ## Estimation of the HSBM with 2 latent groups, random initialization and m-affiliation model
  est <- HSBM(Hypergraph = HG, Q = 2, start = 0, model = 2, tol = 1e-6, maxit_VEM = 100, maxit_FP = 100)
  ## Estimation of the HSBM with 2 latent groups, soft spectral clustering initialization and full model
  est <- HSBM(Hypergraph = HG, Q = 2, start = 2, model = 0, tol = 1e-6, maxit_VEM = 100, maxit_FP = 100)
}
\author{Luca Brusa, Fulvia Pennoni, Francesco Bartolucci}
