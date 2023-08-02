LC_sample <- function (n, piv, Phi, filename = NULL) {
  r <- dim(Phi)[2]
  c <- dim(Phi)[1]
  k <- dim(Phi)[3]
  Y = matrix(0, n, r)
  for (i in 1:n) {
    u = k + 1 - sum(stats::runif(1) < cumsum(piv))
    ind = 0
    for (j in 1:r) {
      ind = ind + 1
      Y[i, ind] = c - sum(stats::runif(1) < cumsum(Phi[, j, u]))
    }
  }
    
  Y <- as.data.frame(Y)
  
  if (is.null(filename)) {
    return(list(Y = Y, piv = piv, Phi = Phi))
  } else {
    filename <- if (grepl("\\.Rdata$", filename) | grepl("\\.RData$", filename)) 
      substr(filename, 1, nchar(filename)-6)
    else filename
    save(Y, piv, Phi, file = paste0(filename, ".RData"))
  }
}



HMcat_sample <- function (n, piv, Pi, Phi, filename = NULL) {
  k = length(piv)
  dd = dim(Phi)
  c = dim(Phi)[1]
  TT = dim(Pi)[3]
  if (length(dd) > 2) {
    r = dd[3]
  } else {
    r = 1
  }
  
  Y = matrix(0, n, TT * r)
  for (i in 1:n) {
    u = k + 1 - sum(runif(1) < cumsum(piv))
    ind = 0
    for (j in 1:r) {
      ind = ind + 1
      Y[i, ind] = c - sum(runif(1) < cumsum(Phi[, u, j]))
    }
    for (t in 2:TT) {
      u = k + 1 - sum(runif(1) < cumsum(Pi[u, , t]))
      for (j in 1:r) {
        ind = ind + 1
        Y[i, ind] = c - sum(runif(1) < cumsum(Phi[, u, j]))
      }
    }
  }
  S = array(t(Y), c(r, TT, n))
  S = aperm(S)
  if (r == 1) {
    S = S[, , 1]
  }
  Y <- as.data.frame(LMest::matrices2long(Y = S))
  
  if (is.null(filename)) {
    return(list(Y = Y, piv = piv, Pi = Pi, Phi = Phi))
  } else {
    filename <- if (grepl("\\.Rdata$", filename) | grepl("\\.RData$", filename)) 
      substr(filename, 1, nchar(filename)-6)
    else filename
    save(Y, piv, Pi, Phi, file = paste0(filename, ".RData"))
  }
}



HMcont_sample <- function (n, piv, Pi, Mu, Si, filename = NULL) {
  if (is.vector(Mu)) {
    r = 1
    k = length(Mu)
    Mu = matrix(Mu, r, k)
  } else {
    r = nrow(Mu)
    k = ncol(Mu)
  }
  TT = dim(Pi)[3]
  if (r == 1) Si = matrix(Si, r, r)
  
  Y = array(0, c(n, TT, r))
  for (i in 1:n) {
    u = k + 1 - sum(runif(1) < cumsum(piv))
    Y[i, 1, ] = mvtnorm:::rmvnorm(1, Mu[, u], Si)
    for (t in 2:TT) {
      u = k + 1 - sum(runif(1) < cumsum(Pi[u, , t]))
      Y[i, t, ] = mvtnorm:::rmvnorm(1, Mu[, u], Si)
    }
  }
  Y <- LMest::matrices2long(Y = Y)
  
  if (is.null(filename)) {
    return(list(Y = Y, piv = piv, Pi = Pi, Mu = Mu, Si = Si))
  } else {
    filename <- if (grepl("\\.Rdata$", filename) | grepl("\\.RData$", filename)) 
      substr(filename, 1, nchar(filename)-6)
    else filename
    save(Y, piv, Pi, Mu, Si, file = paste0(filename, ".RData"))
  }
}



SB_sample <- function (n, piv, B, filename = NULL) {
  k <- length(piv)
  
  true_clusters <- sample(x = 1:k, size = n, replace = TRUE, prob = piv)
  edges <- RcppAlgos::comboGeneral(v = n, m = 2, repetition = TRUE)
  Y <- matrix(0, nrow = n, ncol = n)
  for (i in 1:nrow(edges)) {
    ind1 <- true_clusters[edges[i, ]][1]
    ind2 <- true_clusters[edges[i, ]][2]
    prob <- B[ind1, ind2]
    exist_edge <- rbinom(n = 1, size = 1, prob = prob)
    if (exist_edge == 1) {
      Y[edges[i, 1], edges[i, 2]] <- 1
      Y[edges[i, 2], edges[i, 1]] <- 1
    }
  }
  
  if (is.null(filename)) {
    return(list(Y = Y, true_clusters = true_clusters, piv = piv, B = B))
  } else {
    filename <- if (grepl("\\.Rdata$", filename) | grepl("\\.RData$", filename)) 
      substr(filename, 1, nchar(filename)-6)
    else filename
    save(Y, true_clusters, piv, B, file = paste0(filename, ".RData"))
  }
}


