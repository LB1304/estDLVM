\name{est_SB}
\alias{est_SB}
\title{Estimation of SB model for network data}
\description{
This function performs approximate maximum likelihood estimation of the parameters of the stochastic block model for network data. Two different versions of the variational expectation-maximization algorithm are implemented: standard (VEM) and evolutionary (EVEM).
}
\usage{
est_SB(data, k, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, maxit_FP = 500, sv = NULL, algorithm = c("VEM", "EVEM"), 
       evolutionaryOptions = list(n_parents = NULL, n_children = NULL, prob_mut = NULL, R = NULL))
}
\arguments{
  \item{data}{Either a \code{data.frame} or a \code{matrix} representing the adjacency matrix}
  \item{k}{An integer specifying the number of latent blocks}
  \item{tol_lk, tol_theta}{Tolerance levels for convergence}
  \item{maxit}{Maximum number of iterations for the EM algorithms}
  \item{maxit_FP}{Maximum number of iterations for the fixed point algorithms in the VE step}
  \item{sv}{Starting value for the VEM algorithms; default is \code{NULL}}
  \item{algorithm}{Version of the VEM algorithm employed: \code{"VEM"} for the standard VEM and \code{"EVEM"} for the evolutionary VEM algorithms}
  \item{evolutionaryOptions}{A list containing the options for the evolutionary EM algorithm (see the \emph{Details} section)}
}
\details{
The list \code{evolutionaryOptions} contains the evolutionary constants used to optimally set the EVEM algorithm:
\itemize{
  \item \code{n_parents} number of parents;
  \item \code{n_children} number of offspring;
  \item \code{prob_mut} probability to perform mutation;
  \item \code{R} number of cycles of the standard VEM algorithm in each update step.
}
}
\value{
  \item{LogLik}{Maximum log-likelihood function at convergence of the VEM algorithm}
  \item{J}{Maximum ELBO function at convergence of the VEM algorithm}
  \item{J_vec}{ELBO trace at every step of the VEM algorithm}
  \item{it}{Number of iterations of the EM algorithm}
  \item{piv}{Estimate of the weight vector (k)}
  \item{B}{Estimate of the conditional response probabilities (k x k)}
  \item{k}{Number of latent blocks}
  \item{N_par}{Number of free parameters}
  \item{V}{Matrix of the posterior probabilities for each unit and latent block (n x k)}
  \item{icl}{Value of the Integrated Classification Likelihood Criterion for model selection}
  \item{call}{Command used to call the function}
}
\references{
  See [add paper when available], for more details.
}
\examples{
require(estDLVM)

## Generate sample from the SB model
n <- 500
piv <- runif(3); piv <- piv/sum(piv)
B <- matrix(0.3, ncol = 3, nrow = 3); diag(B) <- 0.7
sample <- SB_sample(n, piv, B)
Y <- sample$Y

## Standard VEM algorithm
stdSB <- est_SB(data = Y, k = 3, algorithm = "VEM")

## Evolutionary VEM algorithm
evoSB <- est_SB(data = Y, k = 3, algorithm = "EVEM", 
                evolutionaryOptions = list(n_parents = 5, n_children = 10, prob_mut = 0.02, R = 10))
}
\author{Luca Brusa, Fulvia Pennoni, Francesco Bartolucci}

