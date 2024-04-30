### Proximal Gradient Descent and Cross Validation for Trace-norm Regularization ### 

import numpy as np
from qics.AEGD_func import norm_scaled
import time
import argparse
import pickle
from pathlib import Path
from numpy.linalg import matrix_rank


parser = argparse.ArgumentParser()
##################################################
# parameters for DGP
parser.add_argument('--outDIR',
                    default='./',
                    type=str,
                    metavar='',
                    help='output directory')
parser.add_argument('--nn',
                    default=100,
                    type=int,
                    metavar='',
                    help='sample size')
parser.add_argument('--pp',
                    default=5,
                    type=int,
                    metavar='',
                    help='number of predictors')
parser.add_argument('--qq',
                    default=5,
                    type=int,
                    metavar='',
                    help='number of responses')
parser.add_argument('--rr',
                    default=2,
                    type=int,
                    metavar='',
                    help='rank of matrix A')
parser.add_argument('--rhoX',
                    default=0.1,
                    type=float,
                    metavar='',
                    help='correlation level in X')
parser.add_argument('--bb',
                    default=0.1,
                    type=float,
                    metavar='',
                    help='SNR level controller')
parser.add_argument('--nrep',
                    default=5,
                    type=int,
                    metavar='',
                    help='number of replicates')

args1 = parser.parse_args()

def pgd(X, Y, alpha=100.0, lr=1e-4, max_iter=10000): 
    W = np.zeros((X.shape[1], Y.shape[1]))
    obj = 0.5 * np.sum((X @ W - Y)**2)
    for _ in range(max_iter):
        grad = X.T @ (X @ W - Y)
        W = W - lr * grad
        U, S, V = np.linalg.svd(W, full_matrices=False)
        S = np.maximum(0, S - lr * alpha)
        W_new = U @ np.diag(S) @ V
        obj_new = 0.5 * np.sum((X @ W_new - Y)**2) + alpha * np.sum(S) 
        if obj_new < obj:
            obj = obj_new
            W = W_new
        else:
            break
    return W


def pgd_cv(X, Y, X_val, Y_val, alphas, lr=1e-4, max_iter=10000, cv=5, seed=0): 
    np.random.seed(seed)
    idx = np.random.permutation(X.shape[0]) #  cv
    alpha_opt = alphas[0]
    err_opt = np.inf
    for alpha in alphas:
        err = 0.0
        for c in range(cv):
            W = pgd(X[idx != c], Y[idx != c], alpha=alpha, lr=lr, max_iter=max_iter) 
            err += np.sqrt(np.sum((X_val @ W - Y_val)**2) / Y_val.size) / cv
            if err < err_opt:
                err_opt = err
                alpha_opt = alpha
    return alpha_opt


nn = args1.nn
pp = args1.pp
qq = args1.qq
rr = args1.rr
bb = args1.bb
rhoX = args1.rhoX
nrep = args1.nrep 

n_val = 2 * nn 
n_test = 2 * nn
mean1 = [0.]*pp
cov1 = np.zeros((pp, pp)) 
for i in range(pp):
    for j in range(pp):
        cov1[i, j] = rhoX**np.abs(i-j)


##################################################
# create output folder
output_folder = (
    f"{args1.outDIR}"
    f"/p{args1.pp}"
    f"q{args1.qq}"
    f"r{args1.rr}"
    f"rX{args1.rhoX}"
    f"b{args1.bb}"
)
Path(output_folder).mkdir(exist_ok=True)
print('Folder %s created!' % output_folder)

rr_est, est_err, fit_err, pred_err = [], [], [], []

# experiment starts
starttime = time.time()
print('Start!')

for seed in range(nrep):
    np.random.seed(seed)
    X_train = np.random.multivariate_normal(mean1, cov1, nn) 
    X_val = np.random.multivariate_normal(mean1, cov1, n_val) 
    X_test = np.random.multivariate_normal(mean1, cov1, n_test)
    mat_B0 = np.random.randn(pp, rr)
    mat_B1 = np.random.randn(rr, qq)
    A = bb * mat_B0 @ mat_B1
    E_train = np.random.randn(nn, qq)
    E_val = np.random.randn(n_val, qq) 
    E_test = np.random.randn(n_test, qq)
    Y_train = X_train @ A + E_train 
    Y_val = X_val @ A + E_val 
    Y_test = X_test @ A + E_test
    alpha_opt = pgd_cv(X_train, Y_train, X_val, Y_val, np.logspace(1, 3, 11), lr=1e-4, max_iter=1000, cv=5, seed=seed) 
    A_est = pgd(X_train, Y_train, alpha=alpha_opt, lr=1e-4, max_iter=1000)
    rr_est.append( matrix_rank(A_est) )
    est_err.append( norm_scaled(A_est, A) )
    fit_err.append( norm_scaled(X_train@A_est, X_train@A) )
    pred_err.append( norm_scaled(X_test@A_est, Y_test) )

print("DONE!")
print(f"That took {time.time() - starttime} seconds")

res_dict = {'Rank': rr_est, 'Est': est_err, 'Fit': fit_err, 'Pred': pred_err}

##################################################
# export results as pickle
output_file = (
    f"{output_folder}"
    f"/[PDG]sim_nn_{args1.nn}"
    f"_pp_{args1.pp}"
    f"_qq_{args1.qq}"
    f"_rr_{args1.rr}"
    f"_rhoX_{args1.rhoX}"
    f"_bb_{args1.bb}"
    f"_nrep_{args1.nrep}"
    f".pkl"
)
print(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(res_dict, f)
print("Data saved!\n")

print('Rank:', f"{np.mean(rr_est):.3f}({np.std(rr_est):.3f})")
print('Est:', f"{np.mean(est_err):.3f}({np.std(est_err):.3f})")
print('Fit:', f"{np.mean(fit_err):.3f}({np.std(fit_err):.3f})")
print('Pred:', f"{np.mean(pred_err):.3f}({np.std(pred_err):.3f})")


