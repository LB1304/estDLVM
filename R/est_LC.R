est_LC <- function (Y, yv, k, tol = 1e-8, maxit = 1e3, sv = NULL, algorithm = c("EM", "TEM", "EEM"), 
                    temperingOptions = list(profile = NULL, alpha = NULL, beta = NULL, rho = NULL, T0 = NULL),
                    evolutionaryOptions = list(n_parents = NULL, n_children = NULL, prob_mut = NULL, R = NULL)) {
  
  # Starting Values
  if (is.null(sv)) {
    sv <- LC_sv(Y = Y, k = k)
  }
  
  # EM/TEM/EEM algorithms
  if (algorithm == "EM") {
    out <- LC_em(Y = Y, yv = yv, k = k, tol = tol, maxit = maxit, piv = sv$piv, Piv = sv$Piv, Psi = sv$Psi, Phi = sv$Phi)
  } else if (algorithm == "TEM") {
    profile <- temperingOptions$profile
    temperingOptions[sapply(temperingOptions, is.null)] <- NULL
    if (is.null(profile) || !(profile %in% c(1, 2, 3))) {
      stop("Specify an available tempering profile.")
    } else {
      profile_pars <- temperingOptions
    }
    out <- LC_tem(Y = Y, yv = yv, k = k, tol = tol, maxit = maxit, piv = sv$piv, Piv = sv$Piv, Psi = sv$Psi, Phi = sv$Phi,
                  profile = profile, profile_pars = profile_pars)
    out$temperingOptions <- temperingOptions
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
    out <- LC_eem(Y = Y, yv = yv, k = k, tol = tol, maxit = maxit, 
                  n_parents = n_parents, n_children = n_children, prob_mut = prob_mut, R = R)
    out$evolutionaryOptions <- evolutionaryOptions
  } else {
    stop("Specify an available algorithm.")
  }
  
  return(out)
}


