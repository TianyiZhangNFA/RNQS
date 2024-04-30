SRS <- function(data_Y, data_X, type = "STRS-MC", rank_X = NULL, self_tune = TRUE) {
  
  # Self-tuning Rank Selection.
  # 
  # args:
  #   data_Y: n by m data matrix of response
  #   data_X: n by p data matrix of features
  #   type: one of "STRS-DB", "STRS-MC" and "SSTRS". 
  #         "SSTRS" is a simpler version when either n >> m or n << m
  #         "STRS-DB" and "STRS-MC" are for general dimensional settings. 
  #         "STRS-DB" uses deterministic expressions for updating lambda
  #         while "STRS-MC" updates lambda by Monte-Carlo simulations. 
  #   rank_X: the specified rank of X. If not specified, estimate from X.
  #   self_tune: TRUE if iteratively estimate the rank and FALSE otherwise.
  #         The default is TRUE.
  #
  # return: the estimated rank
  
  n <- dim(data_Y)[1];  m <- dim(data_Y)[2];  p <- dim(data_X)[2]
  if (is.null(rank_X)) {
    q <- max(which(svd(data_X)$d >= 1e-4))       #//  Estimate the rank of X 
  } else {
    q <- rank_X
  }
  
  losses <- Compute_total_loss(data_Y, data_X)
  P = data_X %*% ginv(t(data_X) %*% data_X) %*% t(data_X)
  
  if (type == "STRS-MC") {
    squarePEs_MC <- MCSquarePEs(q, m, 10)
    resid_MC <- mean(rchisq(10, (n - q) * m))
    
    prev_lambda <- 2.01 * squarePEs_MC[1]
  } else if (type %in% c("STRS-DB", "SSTRS")) {
    prev_lambda <- 2.01 * (sqrt(m) + sqrt(q)) ^ 2
  } else {
    cat("Type must be one of STRS-MC, STRS-DB and SSTRS...")
    stop()
  }

  prev_rank <- Est_rank(losses, prev_lambda, n, q, m)
  
  if (prev_rank != 0 & self_tune) {   
    # iteratively estimate the rank when self_tune is TRUE.
    while(1) {
      curr_lambda <- switch(type, "STRS-DB" = Update_lbd_DB(n, q, m, prev_rank),
                            "STRS-MC" = Update_lbd_MC(n, q, m, prev_rank, 
                                                      squarePEs_MC, resid_MC),
                            "SSTRS" = Update_lbd_DB_simple(n, q, m, prev_rank))
      curr_rank <- Est_rank(losses, curr_lambda, n, q, m, prev_rank)
      if (curr_rank == prev_rank)
        return(curr_rank) 
      prev_rank <- curr_rank
      prev_lambda <- curr_lambda
    }  
  }
  return(prev_rank)
}


