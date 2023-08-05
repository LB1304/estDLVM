est_SB <- function (data, k, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, maxit_FP = 500, sv = NULL, algorithm = c("VEM", "EVEM"), 
                    evolutionaryOptions = list(n_parents = NULL, n_children = NULL, prob_mut = NULL, R = NULL)) {
  
  if (is.data.frame(data) | is.matrix(data)) {
    Y <- as.matrix(data)
  } else {
    stop("Provide data in one of the supported format.")
  }
  
  if (k == 1) {
    out <- SB_k1(Y)
    out$call <- match.call()
    return(out)
  }
  
  # Starting Values
  if (is.null(sv)) {
    sv <- SB_SV(Y = Y, k = k)
  }
  
  # VEM/EVEM algorithms
  if (algorithm == "VEM") {
    out <- SB_VEM(Y = Y, k = k, tol_lk = tol_lk, tol_theta = tol_theta, maxit = maxit, maxit_FP = maxit_FP, V = sv[[1]])
  } else if (algorithm == "EVEM") {
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
  } else {
    stop("Specify an available algorithm.")
  }
  
  out$call <- match.call()
  class(out) <- "SB"
  return(out)
}
