dir1 = './simulation'

source(paste0(dir1,'/STRS.R'))
source(paste0(dir1,'/Helper.R'))


##################################################################
A_solver_real <- function(X, Y, type1 = "STRS-MC") {
    n = nrow(X)
    p = ncol(X)
    q = ncol(X)

    #### Naive OLS estimator of A
    A_OLS = solve(t(X) %*% X) %*% t(X) %*% Y

    #### STRS ####
    # Step 1 Calcualte projection matrix P and symmetric matrix Y'PY
    P = X %*% solve(t(X) %*% X ) %*% t(X)
    YPY= t(Y) %*% P %*% Y

    # Step 2 eigen-decomposition of YPY in descending order
    YPY_eigen = eigen(YPY, symmetric = TRUE)
    lambda = YPY_eigen$values
    V = YPY_eigen$vectors

    # Step 3 Compute W_k and G_k
    W = A_OLS %*% V
    G = t(V)
    r_STRS = SRS(Y, X, type1)
    W_k = matrix(W[,1:r_STRS], nrow=dim(W)[1])
    G_k = matrix(G[1:r_STRS,], ncol=dim(G)[2])

    # Step 4 final estimator
    A_STRS = W_k %*% G_k



    #### Output results ####
    out_list = list("A_OLS" = A_OLS, "A_STRS" = A_STRS,  "r_STRS" = r_STRS)
    return(out_list)
}
