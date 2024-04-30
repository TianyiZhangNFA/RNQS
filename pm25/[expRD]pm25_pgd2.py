### Proximal Gradient Descent and Cross Validation for Trace-norm Regularization ### 

import time
import argparse
import pickle
from pathlib import Path

import numpy as np
from numpy.linalg import matrix_rank
from qics.AEGD_func import norm_scaled

"""
This script implements TR on the pm2.5 real data example. 
"""


parser = argparse.ArgumentParser()

##################################################
# parameters for DGP
parser.add_argument('--inDIR',
                    default='.',
                    type=str,
                    metavar='',
                    help='input directory')
parser.add_argument('--outDIR',
                    default='.',
                    type=str,
                    metavar='',
                    help='output directory')

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


##################################################
# create output folder
output_folder = args1.outDIR
Path(output_folder).mkdir(exist_ok=True)
print('Folder %s created!' % output_folder)


# experiment starts
starttime = time.time()
print('Start!')


##################################
# load data files
##################################
fileX = args1.inDIR + '/Xdata.csv'
fileY = args1.inDIR + '/Ydata.csv'
Xdata = np.loadtxt(fileX, delimiter=',')
Ydata = np.loadtxt(fileY, delimiter=',')
##################################
# sample size of trn/val/test
n_trn = 600
n_test = 128
##################################
# X
X_train = Xdata[:n_trn, :]
X_test = Xdata[-n_test:, :]
# Y
Y_train = Ydata[:n_trn, :]
Y_test = Ydata[-n_test:, :]


alpha_opt = pgd_cv(X_train, Y_train, X_test, Y_test, np.logspace(1, 3, 11), lr=1e-3, max_iter=100, cv=5, seed=10)
A_est = pgd(X_train, Y_train, alpha=alpha_opt, lr=1e-3, max_iter=100)

rr_est = matrix_rank(A_est)
pred_err = norm_scaled(X_test@A_est, Y_test)

print("DONE!")
print(f"That took {time.time() - starttime} seconds")

res_dict = {'Rank': rr_est, 'Pred': pred_err}

##################################################
# export results as pickle
output_file = (
    f"{output_folder}"
    f"/[PDG2]pm25"
    f".pkl"
)
print(output_file)

with open(output_file, 'wb') as f:
    pickle.dump(res_dict, f)
print("Data saved!\n")

print('Rank:', f"{rr_est:.3f}")
print('Pred:', f"{pred_err:.3f}")


