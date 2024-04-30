rm(list = ls())

# set working directory
dir1 = './pm25'
setwd(dir1)

# load STRS
library(MASS)
source(paste0(dir1,'/utl_real.R'))


# norm_scaled function
norm_scaled = function(mat_est, mat_true){
    nrow1 = dim(mat_est)[1]
    ncol1 = dim(mat_est)[2]
    norm1 = norm((mat_est - mat_true), type = "F") / sqrt(nrow1 * ncol1)
    return(norm1)
}

# load X and Y
Xdata = read.csv(paste0(dir1, '/Xdata.csv'), header = F)
Ydata = read.csv(paste0(dir1, '/Ydata.csv'), header = F)

# train-test split
Xtrain = as.matrix(Xdata[1:600,])
Ytrain = as.matrix(Ydata[1:600,])

Xtest = as.matrix(Xdata[601:728,])
Ytest = as.matrix(Ydata[601:728,])

# STRS solver together with OLS
Result = A_solver_real(Xtrain, Ytrain, type1='STRS-MC')

# matrix A estimated by OLS and STRS
A_OLS = Result$A_OLS
A_STRS = Result$A_STRS
Ytest_STRS = Xtest %*% A_STRS
Ytest_OLS = Xtest %*% A_OLS

# rank of STRS
Result$r_STRS

# scaled prediction error
norm_scaled(Ytest_STRS, Ytest)
norm_scaled(Ytest_OLS, Ytest)
