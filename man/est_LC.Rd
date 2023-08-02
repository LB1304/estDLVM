\name{est_LC}
\alias{est_LC}
\title{Estimation of LC model for categorical response variables}
\description{
This function performs maximum likelihood estimation of the parameters of the latent class model for categorical response variables. Three different versions of the expectation-maximization algorithm are implemented: standard (EM), tempered (TEM), and evolutionary (EEM).
}
\usage{
est_LC(data, k, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, sv = NULL, algorithm = c("EM", "TEM", "EEM"), 
       temperingOptions = list(profile = NULL, alpha = NULL, beta = NULL, rho = NULL, T0 = NULL),
       evolutionaryOptions = list(n_parents = NULL, n_children = NULL, prob_mut = NULL, R = NULL))
}
\arguments{
  \item{data}{Either a \code{data.frame} or a \code{matrix}}
  \item{k}{An integer specifying the number of latent classes}
  \item{tol_lk, tol_theta}{Tolerance levels for convergence}
  \item{maxit}{Maximum number of iterations for the EM algorithms}
  \item{sv}{Starting value for the EM algorithms; default is \code{NULL}}
  \item{algorithm}{Version of the EM algorithm employed: \code{"EM"} for the standard EM, \code{"TEM"} for the tempered EM, and \code{"EEM"} for the evolutionary EM algorithms}
  \item{temperingOptions}{A list containing the options for the tempered EM algorithm: type of tempering profile, and values for the tempering constants}
  \item{evolutionaryOptions}{A list containing the options for the evolutionary EM algorithm: number of parents (\code{n_parents}), and number of children (\code{n_children}), probability of mutation (\code{prob_mut}), and number of steps for the EM algorihm during the evolution (\code{R})}
}
\value{
  \item{LogLik}{Maximum log-likelihood function at convergence of the EM algorithm}
  \item{LogLik_vec}{Log-likelihood trace at every step of the EM algorithm}
  \item{it}{Number of iterations of the EM algorithm}
  \item{piv}{Estimate of the weight vector (k)}
  \item{Phi}{Estimate of the conditional response probabilities (C x r x k)}
  \item{k}{Number of latent classes}
  \item{N_par}{Number of free parameters}
  \item{V}{Matrix of the posterior probabilities for each response configuration and latent class (ns x k)}
  \item{aic}{Value of the Akaike Information Criterion for model selection}
  \item{bic}{Value of the Bayesian Information Criterion for model selection}
  \item{call}{Command used to call the function}
}
\references{
  See [add paper when available], for more details.
}
\examples{
require(estDLVM)

## Generate sample from the LC model
n <- 500
piv <- runif(3); piv <- piv/sum(piv)
Phi <- array(c(rep(c(0.7, 0.2, 0.1), 6), rep(c(0.15, 0.7, 0.15), 6), rep(c(0.1, 0.2, 0.7), 6)), dim = c(3, 6, 3))
sample <- LC_sample(n, piv, Phi)
Y <- sample$Y

## Standard EM algorithm
std <- est_LC(data = Y, k = 3, algorithm = "EM")

## Tempered EM algorithm
tmp <- est_LC(data = Y, k = 3, algorithm = "TEM", 
              temperingOptions = list(profile = 3, alpha = 0.8, beta = 20, rho = 90, T0 = 10))

## Evolutionary EM algorithm
evo <- est_LC(data = Y, k = 3, algorithm = "EEM", 
              evolutionaryOptions = list(n_parents = 5, n_children = 10, prob_mut = 0.02, R = 10))
}
\author{Luca Brusa, Fulvia Pennoni, Francesco Bartolucci}

