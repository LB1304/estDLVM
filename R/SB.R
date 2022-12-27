est_SB <- function (Y, k, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, maxit_FP = 500, sv = NULL, algorithm = c("EM", "TEM", "EEM"),
                    temperingOptions = list(profile = NULL, alpha = NULL, beta = NULL, rho = NULL, T0 = NULL),
                    evolutionaryOptions = list(n_parents = NULL, n_children = NULL, prob_mut = NULL, R = NULL)) {
  n <- dim(Y)[1]
  
  # Starting Values
  if (is.null(sv)) {
    sv <- matrix(runif(n*k), nrow = n, ncol = k)
    sv <- sv/rowSums(sv)
  }
  
  # EM/TEM/EEM algorithms
  if (algorithm == "EM") {
    out <- SB_VEM(Y = Y, k = k, n = n, tol_lk = tol_lk, maxit = maxit, maxit_FP = maxit_FP, V = sv)
  } else if (algorithm == "TEM") {
    profile <- temperingOptions$profile
    temperingOptions[sapply(temperingOptions, is.null)] <- NULL
    if (is.null(profile) || !(profile %in% c(1, 2, 3))) {
      stop("Specify an available tempering profile.")
    } else {
      profile_pars <- temperingOptions
    }
    print("Sorry, but this algorithm has not been implemented yet for the stochastic block model.")
  } else if (algorithm == "EEM") {
    evolutionaryOptions[sapply(evolutionaryOptions, is.null)] <- NULL
    if (length(evolutionaryOptions) != 4) {
      stop("All the evolutionary options must be provided.")
    } else {
      n_parents = evolutionaryOptions$n_parents
      n_children = evolutionaryOptions$n_children
      prob_mut = evolutionaryOptions$prob_mut
      R = evolutionaryOptions$R
    }
    out <- SB_EVEM(Y = Y, k = k, tol_lk = tol_lk, tol_theta = tol_theta, maxit = maxit, maxit_FP, 
                   n_parents = n_parents, n_children = n_children, prob_mut = prob_mut, R = R)
    out$evolutionaryOptions <- evolutionaryOptions
  } else {
    stop("Specify an available algorithm.")
  }
  
  return(out)
}