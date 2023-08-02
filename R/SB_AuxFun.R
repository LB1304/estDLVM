SB_SV <- function(Y, k) {
  n <- nrow(Y)
  
  V <- matrix(runif(n*k), nrow = n, ncol = k)
  V <- V/rowSums(V)
  
  return(list(V = V))
}