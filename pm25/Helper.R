Compute_total_loss = function(Y, X) {
  # Compute loss ||Y-PY_r||^2 at different rank r
  # args: Y[n,p] and X[n,p]
  # return: list[[1]]: vector of all losses
  #         list[[2]]: vector of d_j(PY)^2
  if (is.null(X)) {
    svdPYs = svd(Y, nu=0, nv=0)$d
    cumPYs = cumsum(svdPYs^2)
    null = norm(Y, "F")^2
    allLoss = c(null, null-cumPYs)
  } else {
    PY = X %*% ginv(t(X) %*% X) %*% t(X) %*% Y
    svdPYs = svd(PY, nu=0, nv=0)$d
    cumPYs = cumsum(svdPYs^2)
    null = norm(PY, "F")^2
    resid = norm(Y - PY, "F")^2
    allLoss = c(null,null-cumPYs) + resid
  }
  return(list(allLoss=allLoss, PYs=svdPYs^2))
}

Est_rank <- function(losses, lambda, n, q, m, lowerRank = 0) {
  squarePYs <- losses$PYs
  R <- min(q, m, length(squarePYs))
  allLoss <- losses$allLoss 
  counter <- lowerRank
  Rmax <- min(R, trunc((n * m - 1) / lambda))
  while (counter < Rmax) {
    counter <- counter + 1
    tmpLoss <- allLoss[counter] / (n * m - lambda * (counter - 1))
    if (squarePYs[counter] < lambda * tmpLoss)
      return(counter - 1)
  }
  return(Rmax)
}

MCSquarePEs = function(q, m, Nsim = 200) {
  # Use Monte Carlo to estimate d_j(PE)^2, for j = 1,...,q
  N <- min(q, m)
  PEs = matrix(0, N, Nsim)
  for (is in 1:Nsim) {
    PEs[ ,is] = svd(matrix(rnorm(q*m), q, m), nu = 0, nv = 0)$d[1:N] ^ 2
  }
  return(apply(PEs, 1, mean))
}

Compute_Rt = function(L, H, r) {
  # Calculate the deterministic bound of Rt 
  indices1 = 1 : (2 * r)
  indices2 = (2 * r + 1) : L
  bound1 = sum((sqrt(H) + sqrt(L - indices1 + 1))^2)
  bound2 = L * H - sum((sqrt(H) - sqrt(indices2 - 1))^2)
  return(min(bound1, bound2))
}

Update_lbd_MC = function(n, q, m, r, SquarePEs, resid, C = 1.05) {
  # Recalculate lambda based on theselected rnk by MonteCarlo simulating the value of (d_1^2(PE)+...+d_(2r)^2(PE))/r at r
  # args: q, m, r are posive integers and Nsim is the specified number of simulations
  # return: a vector with length equal to min(q,m).
  L = min(q,m)
  denom1 = (resid+sum(SquarePEs)-sum(SquarePEs[1:min(2*r,L)]))/SquarePEs[1]+r
  crit1 = C*n*m/denom1
  if (2 * r <= L) {
    if (2 * r <= L - 2) {
      d1 = SquarePEs[2*r+1]
      d2 = SquarePEs[2*r+2]
    } else {
      d1 = d2 = 0
    }
    denom2 = (resid+sum(SquarePEs)-sum(SquarePEs[1:min(2*r,L)]))/(d1+d2)+r
    crit2 = C*n*m / denom2
  } else {
    crit2 = crit1
  }
  return(max(crit1, crit2))
}

Update_lbd_DB <- function(n, q, m, r, C = 0.01) {
  # Calculate new lambda by using deterministic bound
  L <- min(q, m);  H <- max(q, m);  tmp <- (sqrt(q) + sqrt(m))^2
  if (2 * r >= L) {
    numer <- n * m * tmp
    denom <- (1 - C) * (n - q) * m + r * tmp
  } else {
    Rt <- Compute_Rt(L, H, r)
    Ut <- max(tmp, (sqrt(H) + sqrt(L-2*r))^2 + (sqrt(H) + sqrt(L-2*r-1))^2)
    numer <- n * m * Ut
    denom <- (1 - C) * n * m - Rt + r * Ut
  }
  return(numer / denom)
}

Update_lbd_DB_simple <- function(n, q, m, r, C = 0.01) {
  # Calculate new lambda by using deterministic bound
  L <- min(q, m);  H <- max(q, m); tmp <- (sqrt(m) + sqrt(q))^2
  numer <- n * m 
  if (2 * r >= L) {
    denom <- (1 - C) * (n - q) * m / tmp + r 
  } else {
    denom <- (1 - C) * (n * m / 2 / H - r) + r
  }
  return(numer / denom)
}