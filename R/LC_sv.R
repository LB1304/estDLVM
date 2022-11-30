LC_sv <- function(Y, k) {
  c = max(Y) + 1
  ns = nrow(Y)
  r = ncol(Y)
  
  piv = stats::runif(k)
  piv = piv/sum(piv)
  Piv = rep(1, ns) %o% piv
  Phi = array(stats::runif(c * r * k), c(c, r, k))
  for (u in 1:k) for (j in 1:r) Phi[, j, u] = Phi[, j, u]/sum(Phi[, j, u])
  Psi = matrix(1, ns, k)
  for (u in 1:k) {
    for (j in 1:r) {
      Psi[, u] = Psi[, u] * Phi[Y[, j] + 1, j, u]
    }
  }
  
  return(list(piv = piv, Piv = Piv, Phi = Phi, Psi = Psi))
}
