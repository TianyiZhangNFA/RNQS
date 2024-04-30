rm(list = ls())

# Setwd to your local folder location
dir1 = './simulation'
setwd(dir1)

# Load library and utl codes
library(MASS)
source(paste0(dir1,'/utl_real.R'))

############################################
args = commandArgs(trailingOnly = TRUE)

nn_train = as.numeric(args[1])
nn_test = as.numeric(args[2])
pp = as.numeric(args[3])
qq = as.numeric(args[4])
rr = as.numeric(args[5])
rhoX = as.numeric(args[6])
bb = as.numeric(args[7])
nrep = as.numeric(args[8])
output_dir = as.character(args[9])



##### results ######################
# OLS
ols_results = list()
ols_results$rank_est = rep(qq, nrep)
ols_results$est_err = rep(NA, nrep)
ols_results$fit_err = rep(NA, nrep)
ols_results$test_err = rep(NA, nrep)


# STRS
strs_results = list()
strs_results$rank_est = rep(NA, nrep)
strs_results$est_err = rep(NA, nrep)
strs_results$fit_err = rep(NA, nrep)
strs_results$test_err = rep(NA, nrep)

# norm_scaled function
norm_scaled = function(mat_est, mat_true){
    nrow1 = dim(mat_est)[1]
    ncol1 = dim(mat_est)[2]
    norm1 = norm((mat_est - mat_true), type = "F") / sqrt(nrow1 * ncol1)
    return(norm1)
}
#######################################
##### experiment ######################
cat('Experiment begins...\t')
for (i_rep in 1:nrep){
    if (i_rep %% 50 == 0){
        cat(i_rep, '\t')
    }
    # Data Generating Process
    set.seed(1008)
    # define mean and cov of X
    mean11 = rep(0,pp)
    cov11 = matrix(0, nrow=pp, ncol=pp)
    for (i in 1:pp){
        for (j in 1:pp){
            cov11[i,j] = rhoX^(abs(i-j))
        }
    }

    # generate X
    X_train = MASS::mvrnorm(nn_train, mean11, cov11)
    X_test11 = MASS::mvrnorm(nn_test, mean11, cov11)

    # generate A
    mat_B0 = matrix(rnorm(pp*rr), pp, rr)
    mat_B1 = matrix(rnorm(rr*qq), rr, qq)
    A = bb * mat_B0 %*% mat_B1

    # generate E
    set.seed(i_rep*13+1007)
    E_train = matrix(rnorm(nn_train*qq), nn_train, qq)
    E_test11 = matrix(rnorm(nn_test*qq), nn_test, qq)

    # generate Y
    Y_train = X_train %*% A + E_train
    Y_test11 = X_test11 %*% A + E_test11

    # Estimate coefficient matrix A with OLS and STRS
    # "Type must be one of STRS-MC, STRS-DB and SSTRS"
    Result = A_solver_real(X_train, Y_train, type1='STRS-MC')

    # A estimated by OLS and STRS
    A_OLS = Result$A_OLS
    A_STRS = Result$A_STRS
    
    # if dim(A_STRS) does not match dim(A)
    # make fake copies so that they match
    if (dim(A_STRS)[1] < pp) {
        n_copies1 = pp/dim(A_STRS)[1]
        n_copies1_floor = floor(n_copies1)

        n_copies2 = qq/dim(A_STRS)[2]
        n_copies2_floor = floor(n_copies2)

        if (n_copies1_floor == n_copies1){
            if (n_copies2_floor == n_copies2) {
                A_STRS = matrix(rep(A_STRS, n_copies1*n_copies2), pp, qq, byrow=T)
            } else {
                A_STRS_row = c(rep(A_STRS, n_copies2_floor), A_STRS[1, 1:(qq-n_copies2_floor*dim(A_STRS)[2])])
                A_STRS = matrix(rep(A_STRS_row, n_copies1), pp, qq, byrow=T)
            }
        } else {
            if (n_copies2_floor == n_copies2) {
                A_STRS_col = c(rep(A_STRS, n_copies1_floor), A_STRS[1:(qq-n_copies1_floor*dim(A_STRS)[1]), 1])
                A_STRS = matrix(rep(A_STRS_col, n_copies2), pp, qq)
            }
        }
    }

    # Rank estimated by STRS
    r_STRS = Result$r_STRS

    ###################################################################
    # Prediction on testing sample
    Y_test_OLS = X_test11 %*% A_OLS
    Y_test_STRS = X_test11 %*% A_STRS


    # Calculate multiple measures
    # 1. Rank estimate (Rank)
    strs_results$rank_est[i_rep] = r_STRS

    # 2. Scaled Estimation Error (Est)
    strs_results$est_err[i_rep] = norm_scaled(A_STRS, A)
    ols_results$est_err[i_rep] = norm_scaled(A_OLS, A)

    # 3. Scaled RMSE (Fit)
    strs_results$fit_err[i_rep] = norm_scaled(X_train%*%A_STRS, X_train%*%A)
    ols_results$fit_err[i_rep] = norm_scaled(X_train%*%A_OLS, X_train%*%A)

    # 4. Scaled Prediction RMSE (Pred)
    strs_results$test_err[i_rep] = norm_scaled(Y_test_STRS, Y_test11)
    ols_results$test_err[i_rep] = norm_scaled(Y_test_OLS, Y_test11)
}


##############################################################################
# Save results    ############################################################
##############################################################################
cat("\nSaving results...\n")
output_name = paste0(output_dir, '/nn_', nn_train, '_p_', pp, '_q_', qq, 
                     '_r_', rr, '_rhoX_', rhoX, '_bb_', bb, '_nrep_', nrep, '.RData')

cat(output_name, '\n')



##############################################################################
# Print results   ############################################################
##############################################################################

# Rank
msg1 = paste0("Rank: STRS: ", round(mean(strs_results$rank_est), 3), '(', 
              round(sd(strs_results$rank_est), 3), ')\n')
cat(msg1)
# Est
msg21 = paste0("Est: STRS: ", round(mean(strs_results$est_err), 3), '(', 
              round(sd(strs_results$est_err), 3), ')\n')
cat(msg21)
msg22 = paste0("Est: OLS: ", round(mean(ols_results$est_err), 3), '(', 
              round(sd(ols_results$est_err), 3), ')\n')
cat(msg22)
# Fit
msg31 = paste0("Fit: STRS: ", round(mean(strs_results$fit_err), 3), '(', 
              round(sd(strs_results$fit_err), 3), ')\n')
cat(msg31)
msg32 = paste0("Fit: OLS: ", round(mean(ols_results$fit_err), 3), '(', 
              round(sd(ols_results$fit_err), 3), ')\n')
cat(msg32)
# Pred
msg41 = paste0("Pred: STRS: ", round(mean(strs_results$test_err), 3), '(', 
              round(sd(strs_results$test_err), 3), ')\n')
cat(msg41)
msg42 = paste0("Pred: OLS: ", round(mean(ols_results$test_err), 3), '(', 
              round(sd(ols_results$test_err), 3), ')\n')
cat(msg42)

save.image(output_name)
cat("\nDONE!\n")

