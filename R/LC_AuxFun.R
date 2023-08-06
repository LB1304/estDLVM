aggr_data <- function (data) {
  data = as.matrix(data)
  S = data
  rim = nrow(data)
  nc = ncol(data)
  freq = rep(0, rim)
  label = rep(0, rim)
  j = 0
  label0 = 1:rim
  
  while (rim > 0) {
    j = j + 1
    S[j, ] = data[1, ]
    D = t(matrix(S[j, ], nc, rim))
    ind = which(rowSums(D == data) == nc)
    label[label0[ind]] = j
    freq[j] = length(ind)
    data = as.matrix(data[-ind, ])
    label0 = label0[-ind]
    rim = length(label0)
    if (rim == 1) 
      data = t(data)
  }
  
  S = as.matrix(S[1:j, ])
  freq = freq[1:j]
  
  out = list(S = S, freq = freq, label = label)
}


LC_SV <- function(S, k) {
  c = max(S) + 1
  ns = nrow(S)
  r = ncol(S)
  
  piv = stats::runif(k)
  piv = piv/sum(piv)
  Piv = rep(1, ns) %o% piv
  Phi = array(stats::runif(c * r * k), c(c, r, k))
  for (u in 1:k) for (j in 1:r) Phi[, j, u] = Phi[, j, u]/sum(Phi[, j, u])
  Psi = matrix(1, ns, k)
  for (u in 1:k) {
    for (j in 1:r) {
      Psi[, u] = Psi[, u] * Phi[S[, j] + 1, j, u]
    }
  }
  
  return(list(piv = piv, Piv = Piv, Phi = Phi, Psi = Psi))
}