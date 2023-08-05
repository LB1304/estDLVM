\name{est_HMcat}
\alias{est_HMcat}
\title{Estimation of HM model for categorical response variables}
\description{
This function performs maximum likelihood estimation of the parameters of the hidden Markov model for categorical response variables. Three different versions of the expectation-maximization algorithm are implemented: standard (EM), tempered (TEM), and evolutionary (EEM).
}
\usage{
est_HMcat(data, index, k, modBasic, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, sv = NULL, algorithm = c("EM", "TEM", "EEM"), 
       temperingOptions = list(profile = NULL, alpha = NULL, beta = NULL, rho = NULL, T0 = NULL),
       evolutionaryOptions = list(n_parents = NULL, n_children = NULL, prob_mut = NULL, R = NULL))
}
\arguments{
  \item{data}{Either a \code{data.frame} or a \code{matrix} in long format}
  \item{index}{A character vector with two elements, the names of the unit and the time occasions identifiers, respectively}
  \item{k}{An integer specifying the number of latent states}
  \item{modBasic}{An integer specifying the model on the transition probabilities (0 for time-heterogeneity, 1 for time-homogeneity)}
  \item{tol_lk, tol_theta}{Tolerance levels for convergence}
  \item{maxit}{Maximum number of iterations for the EM algorithms}
  \item{sv}{Starting value for the EM algorithms; default is \code{NULL}}
  \item{algorithm}{Version of the EM algorithm employed: \code{"EM"} for the standard EM, \code{"TEM"} for the tempered EM, and \code{"EEM"} for the evolutionary EM algorithms}
  \item{temperingOptions}{A list containing the options for the tempered EM algorithm: type of tempering profile, and values for the tempering constants}
  \item{evolutionaryOptions}{A list containing the options for the evolutionary EM algorithm: number of parents (\code{n_parents}), and number of children (\code{n_children}), probability of mutation (\code{prob_mut}), and number of steps for the EM algorihm during the evolution (\code{R})}
}
\details{
The list \code{temperingOptions} contains the tempering constants used to optimally set the TEM algorithm: 
\itemize{
  \item \code{profile} defines the type of tempering profile. Three versions are implemented:
  \itemize{
    \item a monotonically decreasing profile (\code{profile = 1}):
    \deqn{\tau_h = \tau_0 \cdot \alpha^{h-1};}
    \item a monotonically decreasing exponential profile (\code{profile = 2}):
    \deqn{\tau_h = 1 + e^{\beta - h/\alpha};}
    \item a non-monotonic profile with oscillations of gradually smaller amplitude (\code{profile = 3}):
    \deqn{\tau_h = \text{tanh}\left(\frac{h}{2\rho}\right) + \left(\tau_0 - \beta \cdot \frac{2\sqrt{2}}{3\pi}\right) \cdot \alpha^{h/\rho} + \beta \cdot \text{sinc}\left(\frac{3\pi}{4} + \frac{h}{\rho}\right);}
  }
  \item \code{alpha}, \code{beta}, \code{rho}, and \code{T0} define the tempering constants \eqn{\alpha}, \eqn{\beta}, \eqn{\rho}, \eqn{\tau_0}, respectively.
}

The list \code{evolutionaryOptions} contains the evolutionary constants used to optimally set the EEM algorithm:
\itemize{
  \item \code{n_parents} number of parents;
  \item \code{n_children} number of offspring;
  \item \code{prob_mut} probability to perform mutation;
  \item \code{R} number of cycles of the standard EM algorithm in each update step.
}
}
\value{
  \item{LogLik}{Maximum log-likelihood function at convergence of the EM algorithm}
  \item{LogLik_vec}{Log-likelihood trace at every step of the EM algorithm}
  \item{it}{Number of iterations of the EM algorithm}
  \item{piv}{Estimate of the initial probability vector (k)}
  \item{Pi}{Estimate of the transition probabilities (k x k x T)}
  \item{Phi}{Estimate of the conditional response probabilities (C x r x k)}
  \item{k}{Number of latent states}
  \item{N_par}{Number of free parameters}
  \item{modBasic}{Model on the transition probabilities (0 for time-heterogeneity, 1 for time-homogeneity)}
  \item{V}{Matrix of the posterior probabilities for each response configuration, latent state, and time occasion (ns x k x T)}
  \item{aic}{Value of the Akaike Information Criterion for model selection}
  \item{bic}{Value of the Bayesian Information Criterion for model selection}
  \item{call}{Command used to call the function}
}
\references{
  See [add paper when available], for more details.
}
\examples{
require(estDLVM)

## Generate sample from the HMcat model
n <- 500
piv <- runif(3); piv <- piv/sum(piv)
Pi <- array(c(0.80, 0.10, 0.05, 0.15, 0.80, 0.15, 0.05, 0.10, 0.80), dim = c(3, 3, 5))
Pi[, , 1] <- 0
Phi <- array(c(0.80, 0.15, 0.05, 0.10, 0.80, 0.10, 0.05, 0.15, 0.80), dim = c(3, 3, 6))
sample <- HMcat_sample(n, piv, Pi, Phi)
Y <- sample$Y

## Standard EM algorithm
stdHMcat <- estDLVM::est_HMcat(data = Y, index = c("id", "time"), k = 3, modBasic = 0, algorithm = "EM")
  
## Tempered EM algorithm
tmpHMcat <- est_HMcat(data = Y, index = c("id", "time"), k = 3, modBasic = 0, algorithm = "TEM", 
                      temperingOptions = list(profile = 2, beta = 0, alpha = 1))

## Evolutionary EM algorithm
evoHMcat <- est_HMcat(data = Y, index = c("id", "time"), k = 3, modBasic = 0, algorithm = "EEM", 
                      evolutionaryOptions = list(n_parents = 5, n_children = 10, prob_mut = 0.02, R = 10))
}
\author{Luca Brusa, Fulvia Pennoni, Francesco Bartolucci}

