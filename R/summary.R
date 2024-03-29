summary.LC <- function(x, ...) {
    cat("\nLatent class model for categorical data\n")
    
    c <- dim(x$Phi)[1]; r <- dim(x$Phi)[2]; k <- x$k
    x$piv <- matrix(x$piv, ncol = 1)
    dimnames(x$piv) <- list("Latent Class" = 1:k, "")
    if (k == 1) {
        x$Phi <- array(x$Phi, c(c, r, 1))
    }
    dimnames(x$Phi) <- list("Category" = 0:(c-1), "Item" = 1:r, "Latent Class" = 1:k)
    cat("\nWeights:\n")
    print(round(x$piv, 4))
    cat("\nConditional response probabilities:\n")
    print(round(x$Phi, 4))
    
    cat("\nConvergence info:\n")
    print(data.frame(LogLik = x$LogLik, AIC = x$aic, BIC = x$bic, N_par = x$N_par, k = x$k, row.names = " "))
}

summary.HMcat <- function(x, ...) {
  cat("\nHidden Markov model for categorical data\n")
  
  c <- dim(x$Phi)[1]; r <- dim(x$Phi)[2]; k <- x$k; TT <- dim(x$V)[3]
  x$piv <- matrix(x$piv, ncol = 1)
  dimnames(x$piv) <- list("Latent State" = 1:k, "")
  if (k == 1) {
    x$Pi <- array(x$Pi, c(k, k, 1))
    TT <- 1
  }
  dimnames(x$Pi) <- list("Latent State" = 1:k, "Latent State" = 1:k, "Time Occasion" = 1:TT)
  if (k == 1) {
    x$Phi <- array(x$Phi, c(c, r, 1))
  }
  dimnames(x$Phi) <- list("Category" = 0:(c-1), "Item" = 1:r, "Latent State" = 1:k)
  cat("\nInitial probabilities:\n")
  print(round(x$piv, 4))
  cat("\nTransition probabilities:\n")
  print(round(x$Pi, 4))
  cat("\nConditional response probabilities:\n")
  print(round(x$Phi, 4))
  
  cat("\nConvergence info:\n")
  print(data.frame(LogLik = x$LogLik, AIC = x$aic, BIC = x$bic, N_par = x$N_par, k = x$k, modBasic = x$modBasic, row.names = " "))
}

summary.HMcont <- function(x, ...) {
  cat("\nHidden Markov model for continuous data\n")
  
  r <- dim(x$Mu)[1]; k <- x$k; TT <- dim(x$V)[3]
  x$piv <- matrix(x$piv, ncol = 1)
  dimnames(x$piv) <- list("Latent State" = 1:k, "")
  if (k == 1) {
    x$Pi <- array(x$Pi, c(k, k, 1))
    TT <- 1
  }
  dimnames(x$Pi) <- list("Latent State" = 1:k, "Latent State" = 1:k, "Time Occasion" = 1:TT)
  dimnames(x$Mu) <- list("Item" = 1:r, "Latent State" = 1:k)
  dimnames(x$Si) <- list("Item" = 1:r, "Item" = 1:r)
  cat("\nInitial probabilities:\n")
  print(round(x$piv, 4))
  cat("\nTransition probabilities:\n")
  print(round(x$Pi, 4))
  cat("\nMean vectors:\n")
  print(round(x$Mu, 4))
  cat("\nVariance-covariance matrix:\n")
  print(round(x$Si, 4))
  
  cat("\nConvergence info:\n")
  print(data.frame(LogLik = x$LogLik, AIC = x$aic, BIC = x$bic, N_par = x$N_par, k = x$k, row.names = " "))
}

summary.SB <- function(x, ...) {
  cat("\nStochastic block for network data\n")
  
  k <- length(x$piv)
  x$piv <- matrix(x$piv, ncol = 1)
  dimnames(x$piv) <- list("Latent Block" = 1:k, "")
  if (k == 1) {
    x$B <- matrix(x$B, ncol = k, nrow = k)
  }
  dimnames(x$B) <- list("Latent Block" = 1:k, "Latent Block" = 1:k)
  cat("\nWeights:\n")
  print(round(x$piv, 4))
  cat("\nConnection probabilities:\n")
  print(round(x$B, 4))
  
  cat("\nConvergence info:\n")
  print(data.frame(LogLik = x$LogLik, J = x$J, ICL = x$icl, N_par = x$N_par, k = x$k, row.names = " "))
}
