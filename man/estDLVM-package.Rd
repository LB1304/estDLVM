\name{estDLVM Package}
\alias{estDLVM Package}
\docType{package}
\title{Overview of the package estDLVM}

\description{\packageDescription{estDLVM}}

\details{
The main functions of the package are listed hereafter:
\itemize{
  \item \code{\link{est_LC}}: performs maximum likelihood estimation of the parameters of the latent class model for categorical data using EM, TEM, or EEM algorithms.
  \item \code{\link{est_HMcat}}: performs maximum likelihood estimation of the parameters of the latent class model for categorical longitudinal data using EM, TEM, or EEM algorithms. 
  \item \code{\link{est_HMcont}}: performs maximum likelihood estimation of the parameters of the latent class model for continuous longitudinal data using EM, TEM, or EEM algorithms. 
  \item \code{\link{est_SB}}: performs approximate maximum likelihood estimation of the parameters of the latent class model for network data using VEM or EVEM algorithms. 
}
}

\author{
Luca Brusa [aut, cre], Francesco Bartolucci [aut], Fulvia Pennoni [aut]

\emph{Maintainer}: Luca Brusa (luca.brusa@unimib.it)
}

\references{
Bartolucci, F., Farcomeni, A., Pennoni, F. (2013). \emph{Latent Markov Models for Longitudinal Data}, Chapman and Hall/CRC press.

Bartolucci, F., Bacci, S., and Gnaldi, M. (2014). MultiLCIRT: An R package for multidimensional latent class item response models. \emph{Comput. Stat. Data Anal.}, \bold{71}, 971--985.

Bartolucci, F., Pandolfi, S., and Pennoni, F. (2017). LMest: An R package for latent markov models for longitudinal categorical data. \emph{J. Stat. Softw.}, \bold{81}, 1--38.

Bartolucci, F., Pandolfi, S., and Pennoni, F. (2022). Discrete latent variable models. \emph{Annu. Rev. Stat. Appl.}, \bold{6}, 1--31

Brusa, L., Bartolucci, F., Pennoni, F. (2022). Tempered expectation-maximization algorithm for the estimation of discrete latent variable models. \emph{Comput. Stat.}, \bold{38}, 1--34.

Daudin, J.-J., Picard, F., and Robin, S. (2008). A mixture model for random graphs. \emph{Stat. Comput.}, \bold{18}, 173--183.
}

