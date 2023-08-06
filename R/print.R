Get_opt <- function(x, alg) {
  opt <- as.character(x$call[which(names(x$call) == alg)])
  opt <- gsub("(?<=\\()[^()]*(?=\\))(*SKIP)(*F)|.", "", opt, perl=T)
  opt <- strsplit(opt, ", ")[[1]]
  opt <- strsplit(opt, " = ")
  opt <- matrix(unlist(opt), ncol = length(opt))
  opt_df <- data.frame(t(opt[2, ]))
  colnames(opt_df) <- opt[1, ]
  return(opt)
}

#'@export
print.LC <- function(x, ...) {
  cat("\nLatent class model for categorical data\n")
  
  if ("evolutionaryOptions" %in% names(x$call)) {
    opt <- Get_opt(x = x, alg = "evolutionaryOptions")
    cat("\nEstimation algorithm: EEM algorithm\n")
    cat("\nEvolutionary constants:\n")
    print(opt_df, row.names = FALSE)
  } else if ("temperingOptions" %in% names(x$call)) {
    opt <- Get_opt(x = x, alg = "temperingOptions")
    cat("\nEstimation algorithm: TEM allgorithm\n")
    cat("\nTempering constants:\n")
    print(opt_df, row.names = FALSE)
  } else {
    cat("\nEstimation algorithm: EM algorithm\n")
  }
  
  cat("\nAvailable objects:\n")
  print(names(x))
  
  cat("\nConvergence info:\n")
  print(data.frame(LogLik = x$LogLik, AIC = x$aic, BIC = x$bic, N_par = x$N_par, k = x$k, modBasic = x$modBasic, row.names = " "))
}

#'@export
print.HMcat <- function(x, ...) {
  cat("\nHidden Markov model for categorical data\n")
  
  if ("evolutionaryOptions" %in% names(x$call)) {
    opt <- Get_opt(x = x, alg = "evolutionaryOptions")
    cat("\nEstimation algorithm: EEM algorithm\n")
    cat("\nEvolutionary constants:\n")
    print(opt_df, row.names = FALSE)
  } else if ("temperingOptions" %in% names(x$call)) {
    opt <- Get_opt(x = x, alg = "temperingOptions")
    cat("\nEstimation algorithm: TEM allgorithm\n")
    cat("\nTempering constants:\n")
    print(opt_df, row.names = FALSE)
  } else {
    cat("\nEstimation algorithm: EM algorithm\n")
  }
  
  cat("\nAvailable objects:\n")
  print(names(x))
  
  cat("\nConvergence info:\n")
  print(data.frame(LogLik = x$LogLik, AIC = x$aic, BIC = x$bic, N_par = x$N_par, k = x$k, modBasic = x$modBasic, row.names = " "))
}

#'@export
print.HMcont <- function(x, ...) {
  cat("\nHidden Markov model for continuous data\n")
  
  if ("evolutionaryOptions" %in% names(x$call)) {
    opt <- Get_opt(x = x, alg = "evolutionaryOptions")
    cat("\nEstimation algorithm: EEM algorithm\n")
    cat("\nEvolutionary constants:\n")
    print(opt_df, row.names = FALSE)
  } else if ("temperingOptions" %in% names(x$call)) {
    opt <- Get_opt(x = x, alg = "temperingOptions")
    cat("\nEstimation algorithm: TEM allgorithm\n")
    cat("\nTempering constants:\n")
    print(opt_df, row.names = FALSE)
  } else {
    cat("\nEstimation algorithm: EM algorithm\n")
  }
  
  cat("\nAvailable objects:\n")
  print(names(x))
  
  cat("\nConvergence info:\n")
  print(data.frame(LogLik = x$LogLik, AIC = x$aic, BIC = x$bic, N_par = x$N_par, k = x$k, row.names = " "))
}

#'@export
print.SB <- function(x, ...) {
  cat("\nStochastic block for network data\n")
  
  if ("evolutionaryOptions" %in% names(x$call)) {
    opt <- Get_opt(x = x, alg = "evolutionaryOptions")
    cat("\nEstimation algorithm: EVEM algorithm\n")
    cat("\nEvolutionary constants:\n")
    print(opt_df, row.names = FALSE)
  } else {
    cat("\nEstimation algorithm: VEM algorithm\n")
  }
  
  cat("\nAvailable objects:\n")
  print(names(x))
  
  cat("\nConvergence info:\n")
  print(data.frame(LogLik = x$LogLik, J = x$J, ICL = x$icl, N_par = x$N_par, k = x$k, row.names = " "))
}
