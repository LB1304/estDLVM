est_HMcont <- function (data, index, k, modBasic, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, sv = NULL, algorithm = c("EM", "TEM", "EEM"), 
                        temperingOptions = list(profile = NULL, alpha = NULL, beta = NULL, rho = NULL, T0 = NULL), 
                        evolutionaryOptions = list(n_parents = NULL, n_children = NULL, prob_mut = NULL, R = NULL)) {
  
  if (is.data.frame(data) | is.matrix(data)) {
    id.which <- which(names(data) == index[1])
    tv.which <- which(names(data) == index[2])
    id <- data[, id.which]
    tv <- data[, tv.which]
    Y <- data[, -c(id.which, tv.which), drop = FALSE]
    Y_names <- colnames(Y)
    
    S <- HMcont_long2matrices(Y = Y, id = id, time = tv)
    dimnames(S)[[3]] <- Y_names
  } else if (is.list(data) | identical(names(data), c("S", "yv"))) {
    S <- data$S
    yv <- data$yv
  } else {
    stop("Provide data in one of the supported format.")
  }
  
  # Starting Values
  if (is.null(sv)) {
    sv <- HMcont_SV(S = S, k = k)
  }
  
  # EM/TEM/EEM algorithms
  if (algorithm == "EM") {
    out <- HMcont_EM(S = S, k = k, tol_lk = tol_lk, tol_theta = tol_theta, maxit = maxit, piv = sv$piv, Pi = sv$Pi, Mu = sv$Mu, Si = sv$Si, modBasic = modBasic)
  } else if (algorithm == "TEM") {
    profile <- temperingOptions$profile
    temperingOptions[sapply(temperingOptions, is.null)] <- NULL
    if (is.null(profile) || !(profile %in% c(1, 2, 3))) {
      stop("Specify an available tempering profile.")
    } else {
      profile_pars <- temperingOptions
    }
    out <- HMcont_TEM(S = S, k = k, tol_lk = tol_lk, tol_theta = tol_theta, maxit = maxit, piv = sv$piv, Pi = sv$Pi, Mu = sv$Mu, Si = sv$Si, modBasic = modBasic, 
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
    out <- HMcont_EEM(S = S, k = k, tol_lk = tol_lk, tol_theta = tol_theta, maxit = maxit, modBasic = modBasic, 
                      n_parents = n_parents, n_children = n_children, prob_mut = prob_mut, R = R)
    out$evolutionaryOptions <- evolutionaryOptions
  } else {
    stop("Specify an available algorithm.")
  }
  
  return(out)
}