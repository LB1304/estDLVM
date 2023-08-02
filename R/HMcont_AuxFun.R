HMcont_long2matrices <- function (Y, id, time) {
  Y = as.matrix(Y)
  ny = ncol(Y)
  TT = max(time)
  idu = unique(id)
  n = length(idu)
  
  S = array(NA, c(n, TT, ny))
  for (i in 1:n) {
    ind = which(id == idu[i])
    tmp = 0
    for (t in time[ind]) {
      tmp = tmp + 1
      S[i, t, ] = Y[ind[tmp], ]
    }
  }
  
  return(S)
}


HMcont_SV <- function(S, k) {
  
  sY = dim(S)
  n = sY[1]
  TT = sY[2]
  if (length(sY) == 2) {
    r = 1
    if (is.matrix(S))
      S = array(S, c(dim(S), 1))
  } else {
    r = sY[3]
  }
  Yv <- matrix(S, n * TT, r)
  
  Mu = matrix(0, r, k)
  mu = colMeans(Yv, na.rm = TRUE)
  Si = cov(Yv, use = "complete.obs")
  for (u in 1:k) Mu[, u] = mvtnorm:::rmvnorm(1, mu, Si)
  Pi = array(stats::runif(k^2 * TT), c(k, k, TT))
  for (t in 2:TT) Pi[, , t] = diag(1/rowSums(Pi[, , t])) %*% Pi[, , t]
  Pi[, , 1] = 0
  piv = stats::runif(k)
  piv = piv/sum(piv)
  
  return(list(piv = piv, Pi = Pi, Mu = Mu, Si = Si))
}