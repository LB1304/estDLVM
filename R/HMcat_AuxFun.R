HMcat_long2matrices <- function (Y, id, time) {
  idu = unique(id)
  n = length(idu)
  TT = max(time)
  
  Y = as.matrix(Y)
  ny = ncol(Y)
  
  temp <- data.frame(id = id, time = time, Y, check.names = FALSE)
  temp.wide <- reshape(temp, idvar = "id", timevar = "time", direction = "wide")
  temp.wide[is.na(temp.wide)] <- 999
  aggr <- MultiLCIRT::aggr_data(temp.wide[, -1], fort = TRUE)
  temp <- data.frame(1:nrow(aggr$data_dis), aggr$data_dis)
  colnames(temp) <- c("id", attributes(temp.wide)$reshapeWide$varying)
  freq <- aggr$freq
  Y <- stats::reshape(temp, direction = "long", idvar = "id", varying = 2:ncol(temp), sep = ".")
  Y[, -c(1, 2)][Y[, -c(1, 2)] == 999] <- NA
  id <- Y[, 1]
  time <- Y[, 2]
  Y = as.matrix(Y[, -c(1, 2)])
  
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
  
  return(list(S = S, freq = freq))
}


HMcat_SV <- function(S, k) {
  
  sS = dim(S)
  ns = sS[1]
  TT = sS[2]
  if (length(sS) == 2) {
    r = 1
    if (is.matrix(S)) 
      S = array(S, c(sS, 1))
  } else {
    r = sS[3]
  }
  Sv = matrix(S, ns * TT, r)
  bv = apply(Sv, 2, max)
  b = max(bv)
  
  piv = stats::runif(k)
  piv = piv/sum(piv)
  
  Pi = array(stats::runif(k^2 * TT), c(k, k, TT))
  for (t in 2:TT) Pi[, , t] = diag(1/rowSums(Pi[, , t])) %*% Pi[, , t]
  Pi[, , 1] = 0
  
  Phi = array(NA, c(b + 1, k, r))
  for (j in 1:r) {
    Phi[1:(bv[j] + 1), , j] = matrix(stats::runif((bv[j] + 1) * k), bv[j] + 1, k)
    for (u in 1:k) Phi[1:(bv[j] + 1), u, j] = Phi[1:(bv[j] + 1), u, j]/sum(Phi[1:(bv[j] + 1), u, j])
  }
  
  return(list(piv = piv, Pi = Pi, Phi = Phi))
}