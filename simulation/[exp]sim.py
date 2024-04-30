######################################################
# load base packages
######################################################
import time
import argparse
import pickle
# index-based item extraction from an iterable where each element is a tuple of multiple elements
from operator import itemgetter
from functools import partial
import multiprocessing
from itertools import product, combinations, chain
from pathlib import Path
######################################################
# load algorithmic modules
######################################################
import math
import numpy as np
from qics.AEGD_func import AEGD, obj_val, norm_scaled #, loss2obj  # numpy included
from qics.QICS_func import qSearch, pow2list  # random, math, Counter included


"""
This script is the sole script for the simulation. 

Specifically, given the search space consisting of combinations of q-choose-stop_r columns of Y, the script calculate objective values in parallel (multiprocessing.Pool).
"""


######################################################
# read arguments
######################################################

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
parser.add_argument('--tol',
                    default=1e-6,
                    type=float,
                    metavar='',
                    help='tolerance')
parser.add_argument('--num_itr',
                    default=100,
                    type=int,
                    metavar='',
                    help='number of iterations')
parser.add_argument('--stop_rr',
                    default=3,
                    type=int,
                    metavar='',
                    help='upper limit of rank search')
parser.add_argument('--const_qq',
                    default=2,
                    type=int,
                    metavar='',
                    help='a qq-related multiplier in penalty function')
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
parser.add_argument('--ncores',
                    default=5,
                    type=int,
                    metavar='',
                    help='number of cores')
parser.add_argument('--subID',
                    default=0,
                    type=int,
                    metavar='',
                    help='ID for replication purpose or subprocess')
##################################################
# parameters for quantum search algorithm
parser.add_argument('--const_num_itr',
                    default=3,
                    type=int,
                    metavar='',
                    help='multiplier for calculating number of iterations')
parser.add_argument('--kk',
                    default=5,
                    type=int,
                    metavar='',
                    help='number of chains')
parser.add_argument('--gamma',
                    default=0.5,
                    type=float,
                    metavar='',
                    help='learning rate')
args1 = parser.parse_args()



####################################################################
# worker function for DGP
def worker_func_DGP(args1, i_itr):
    """
    AEGD given the indices in A1(inclded in i_itr)
    
    Input:
    args1: user-defined parameters
    i_itr: ith-iterable, product of two sub-iterables, (idx_A1, i_rep), e.g. ((0,4), 0)

    Return: 
    rr_est: rank estimate
    obj_est: obj value
    loss_est: loss value
    est_err: RMSE of A
    fit_err: RMSE of XA
    val_err: RMSE on val set
    test_err: RMSE on test set
    idx_A1: support of A1
    """

    idx_A1, i_rep = i_itr
    # idx_A1, i_rep = ((0,4), 0)
    ##################################
    # DGP
    ##################################
    # np.random.seed(i_rep*100 + args1.subID*37) # random design
    np.random.seed(args1.subID*37+1003)  # fix design
    # define mean and cov of X
    mean1 = [0.]*args1.pp
    cov1 = np.zeros((args1.pp, args1.pp))
    for i in range(args1.pp):
        for j in range(args1.pp):
            cov1[i, j] = args1.rhoX**np.abs(i-j)
    ##################################
    # sample size of validation/testing set
    # n_val = args1.nn
    # n_test = args1.nn
    n_val = 2*args1.nn
    n_test = 2*args1.nn
    ##################################
    # generate X
    X_train = np.random.multivariate_normal(mean1, cov1, args1.nn)
    X_val = np.random.multivariate_normal(mean1, cov1, n_val)
    X_test = np.random.multivariate_normal(mean1, cov1, n_test)
    # generate A
    mat_B0 = np.random.randn(args1.pp, args1.rr)
    mat_B1 = np.random.randn(args1.rr, args1.qq)
    A = args1.bb * mat_B0 @ mat_B1
    # generate E
    np.random.seed(i_rep*17+7+args1.subID*300)
    E_train = np.random.randn(args1.nn, args1.qq)
    E_val = np.random.randn(n_val, args1.qq)
    E_test = np.random.randn(n_test, args1.qq)
    # generate Y
    Y_train = X_train @ A + E_train
    Y_val = X_val @ A + E_val
    Y_test = X_test @ A + E_test
    # rank estimate
    rr_est = len(idx_A1)
    #########################################
    # do a single OLS when A1 = A
    # do GD estimation as usual otherwise
    #########################################
    if rr_est == args1.qq:
        # OLS estimation
        A_est, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)
    else:
        # all columns in matrix A
        idx_A = range(args1.qq)
        idx_A2 = list(set(idx_A) - set(idx_A1))
        # generate Y1, Y2
        Y1 = Y_train[:, idx_A1]
        Y2 = Y_train[:, idx_A2]
        # learning rate
        lr = 10.**-(np.log10(args1.nn) + 2)
        ##################################
        # AEGD-Estimation
        ##################################
        est_AEGD = AEGD()
        _, _, _, A_est = est_AEGD.est_GD(
            X_train, Y1, Y2, lr, args1.tol, args1.num_itr)
    ##################################
    # Calculate objective value and loss value
    obj_est, loss_est, _ = obj_val(
        Y_train - X_train@A_est, rr_est, args1.const_qq)
    ##################################
    # Calculate error-based measures
    est_err = norm_scaled(A_est, A)
    fit_err = norm_scaled(X_train@A_est, X_train@A)
    val_err = norm_scaled(X_val@A_est, Y_val)
    test_err = norm_scaled(X_test@A_est, Y_test)
    return rr_est, obj_est, loss_est, est_err, fit_err, val_err, test_err, idx_A1, i_rep


####################################################################
# worker function for Minimal Finding(MF)
def worker_func_MF(args1, qSearch, tuples_ALL, supList, oracle_idx, i_rep):
    """find the minimum in parallel

    Args:
        args1 (class): input parameters
        qSearch (class): quantum search module
        tuples_ALL (tuple): a BIG tuple that contains all calculated values
        supList (list): a list of support of A1
        oracle_idx (int): the index of the minimal
        i_rep (int): ith-iteration

    Returns:
        idxES(int): true indices of the minimal
        idxQICS(int): indices of the minimal by QICS
        idxFA (int): indices of the minimal by FA
    """
    ##################################
    # tuple_ALL explained (more in [expGen]aos11.py)
    ##################################
    # example/structure of an element of tuple_ALL
    # (1,                  [0] rr_est: rank/support size of A1 (Rank)
    # 2085.068328245891,   [1] obj_est: objective value to to tune FA
    # 2066.1367596765667,  [2] loss_est: raw loss value
    # 0.28105713578943936, [3] est_err: scaled estimation error (Est)
    # 0.8995122769389,     [4] fit_err: scaled RMSE (Fit)
    # 1.338532889385699,   [5] val_err: scaled validation RMSE to tune QICS
    # 1.3650893500891532,  [6] test_err: scaled prediction RMSE (Pred)
    # (0,),                [7] idx_A1: support of A1
    # 0)                   [8] i_rep
    # rr_est, obj_est, loss_est, est_err, fit_err, val_err, test_err, idx_A1, i_rep

    ##################################
    # Process the iterable (i_rep)
    ##################################
    # find the validation error list for QICS and ground truth
    pred_list = [i[5] for i in tuples_ALL if i[8] == i_rep]
    lenPredList = len(pred_list)
    # find objective value list for FA
    obj_list = [i[1] for i in tuples_ALL if i[8] == i_rep]
    ##################################
    # Setup prerequsites for QICS
    ##################################
    # augment pred_list to length of a multiple of power 2
    powOutput = math.ceil(math.log2(lenPredList))
    LOSS_LIST_AG = pow2list(pred_list, powOutput)
    # set number of iterations for QICS
    num_itr = math.ceil(args1.const_num_itr*2*math.log2(powOutput))
    # initialize QICS module and random seed
    qsearch = qSearch(2**powOutput)
    seed = 17*i_rep+3
    ##################################
    # Execution
    ##################################
    # ground truth(ES)
    idxES = LOSS_LIST_AG.index(min(LOSS_LIST_AG))
    # QICS
    idxQICS, _ = qsearch.qAdaSearch(
        LOSS_LIST_AG, num_itr, kk=args1.kk, oracle_idx=oracle_idx, 
        seed=seed, gamma=args1.gamma)
    # FA
    idxFA, _, _ = qsearch.FA(obj_list, supList, args1.stop_rr)
    ##################################
    return idxES, idxQICS, idxFA



def main(): 
    #####################################################
    #                                                   #
    #           Data Generating Process(DGP)            #
    #                                                   #
    #####################################################
    ##################################################
    # create output folder
    output_folder = (
        f"{args1.outDIR}"
        f"/p{args1.pp}"
        f"q{args1.qq}"
        f"r{args1.rr}"
        f"rp{args1.nrep}"
        f"n{args1.nn}"
    )
    Path(output_folder).mkdir(exist_ok=True)
    print('Folder %s created!' % output_folder)
    output_folder = f"{output_folder}/r{args1.rhoX}b{args1.bb}id{args1.subID}"
    Path(output_folder).mkdir(exist_ok=True)
    print('Folder %s created!' % output_folder)
    print('\n===>DGP')
    ##################################################
    # 2020.5.21 iterable using tuple (memory cost: tuple<list<set)
    itrs = (combinations(range(args1.qq), i) for i in range(1, args1.stop_rr+1))
    itr_chain = chain.from_iterable(itrs)
    # create range of replicates
    range_rep = range(args1.nrep)
    itr_nrep = product(itr_chain, range_rep)
    ##################################################
    # pool object
    with multiprocessing.Pool(args1.ncores) as mp:
        # reduce the number of arugments via partial function
        part_func_DGP = partial(worker_func_DGP, args1)
        # experiment starts
        starttime = time.time()
        print('Start!')
        res_tuples_DGP = mp.map(part_func_DGP, itr_nrep)
        # experiment ends
        print("DONE!")
    print(f"That took {time.time() - starttime} seconds")
    ##################################################
    # export DGP values as pickle
    output_file_DGP = (
        f"{output_folder}"
        f"/[AEGD]sim_nn_{args1.nn}"
        f"_pp_{args1.pp}"
        f"_qq_{args1.qq}"
        f"_rr_{args1.rr}"
        f"_stop_rr_{args1.stop_rr}"
        f"_const_qq_{args1.const_qq}"
        f"_rhoX_{args1.rhoX}"
        f"_bb_{args1.bb}"
        f"_nrep_{args1.nrep}"
        f"_ncores_{args1.ncores}"
        f"_subID_{args1.subID}.pkl"
    )
    print(output_file_DGP)
    with open(output_file_DGP, 'wb') as f:
        pickle.dump(res_tuples_DGP, f)
    print("Data saved!\n")

    #####################################################
    #                                                   #
    #           Minimal Finding(MF)                     #
    #                                                   #
    #####################################################


    print('\n===>Minimal Finding')
    ##########################################################
    ## i[7]: Support of idx_A1
    ## i[8]: Each element in iterable repeat nrep times: e.g.
    ## [0,1,2,...,nrep-1,0,1,...,nrep-1,...,nrep-1]
    supList = [i[7] for i in res_tuples_DGP if i[8] == 0]
    # iterable for multiprocessing
    rep_loop = range(args1.nrep)
    ##########################################################
    ## access metric values using the indices in a tuple via itemgetter
    ## 0: rank estimate (Rank)
    ## 3: scaled estimation error (Est)
    ## 4: scaled RMSE (Fit)
    ## 6: scaled prediction RMSE (Pred)
    getter1 = itemgetter(0, 3, 4, 6)
    ##########################################################
    # evaluate ES/QICS/FA
    # oracle index for QICS
    oracle_idx = None
    # pool object
    with multiprocessing.Pool(args1.ncores) as mp:
        # reduce the numper of arugments via partial function
        part_func_MF = partial(worker_func_MF, args1, qSearch,
                            res_tuples_DGP, supList, oracle_idx)
        # experiment starts
        starttime = time.time()
        print('Start!')
        res_tuples_MF = mp.map(part_func_MF, rep_loop)
        # experiment ends
        print("DONE!")
    print(f"That took {time.time() - starttime} seconds\n")
    ##########################################################
    # export minimal findings as pickle
    output_file_MF = (
        f"{output_folder}"
        f"/[MF]sim_nn_{args1.nn}"
        f"_pp_{args1.pp}"
        f"_qq_{args1.qq}"
        f"_rr_{args1.rr}"
        f"_stop_rr_{args1.stop_rr}"
        f"_const_qq_{args1.const_qq}"
        f"_rhoX_{args1.rhoX}"
        f"_bb_{args1.bb}"
        f"_nrep_{args1.nrep}"
        f"_ncores_{args1.ncores}"
        f"_subID_{args1.subID}"
        f"_kk_{args1.kk}"
        f"_const_num_itr_{args1.const_num_itr}"
        f"_gamma_{args1.gamma}.pkl"
    )
    print(output_file_MF)
    with open(output_file_MF, 'wb') as f:
        pickle.dump(res_tuples_MF, f)
    print('Data saved!\n')
    ##########################################################
    # calculate measures
    ##########################################################
    # calculate coverage percentage
    qAda_pct = [1 if idx_i == truth_i else 0 for truth_i,
                idx_i, _ in res_tuples_MF]
    print(f"QICS coverage_pct: {np.mean(qAda_pct):.3f}%\n")
    # create a dict of metrics
    dict_metric = {'rk': np.zeros((args1.nrep, 3)),
                   'est': np.zeros((args1.nrep, 3)),
                   'fit': np.zeros((args1.nrep, 3)),
                   'pred': np.zeros((args1.nrep, 3))
                   }
    for i, (iES, iQICS, iFA) in enumerate(res_tuples_MF):
        # ES
        dict_metric['rk'][i, 0], dict_metric['est'][i, 0], dict_metric['fit'][i,0], dict_metric['pred'][i, 0] = getter1(res_tuples_DGP[i+iES*args1.nrep])
        # QICS
        dict_metric['rk'][i, 1], dict_metric['est'][i, 1], dict_metric['fit'][i,1], dict_metric['pred'][i, 1] = getter1(res_tuples_DGP[i+iQICS*args1.nrep])
        # FA
        dict_metric['rk'][i, 2], dict_metric['est'][i, 2], dict_metric['fit'][i,2], dict_metric['pred'][i, 2] = getter1(res_tuples_DGP[i+iFA*args1.nrep])
    print((
        f"Criterion: ES,QICS,FA mean(std)\n"
        # Rank
        f"Rank: ES: {dict_metric['rk'][:,0].mean(axis=0):.3f}({dict_metric['rk'][:,0].std(axis=0):.3f})\n"
        f"Rank: QICS: {dict_metric['rk'][:,1].mean(axis=0):.3f}({dict_metric['rk'][:,1].std(axis=0):.3f})\n"
        f"Rank: FA: {dict_metric['rk'][:,2].mean(axis=0):.3f}({dict_metric['rk'][:,2].std(axis=0):.3f})\n"
        # Est
        f"Est: ES: {dict_metric['est'][:,0].mean(axis=0):.3f}({dict_metric['est'][:,0].std(axis=0):.3f})\n"
        f"Est: QICS: {dict_metric['est'][:,1].mean(axis=0):.3f}({dict_metric['est'][:,1].std(axis=0):.3f})\n"
        f"Est: FA: {dict_metric['est'][:,2].mean(axis=0):.3f}({dict_metric['est'][:,2].std(axis=0):.3f})\n"
        # Fit
        f"Fit: ES: {dict_metric['fit'][:,0].mean(axis=0):.3f}({dict_metric['fit'][:,0].std(axis=0):.3f})\n"
        f"Fit: QICS: {dict_metric['fit'][:,1].mean(axis=0):.3f}({dict_metric['fit'][:,1].std(axis=0):.3f})\n"
        f"Fit: FA: {dict_metric['fit'][:,2].mean(axis=0):.3f}({dict_metric['fit'][:,2].std(axis=0):.3f})\n"
        # Pred
        f"Pred: ES: {dict_metric['pred'][:,0].mean(axis=0):.3f}({dict_metric['pred'][:,0].std(axis=0):.3f})\n"
        f"Pred: QICS: {dict_metric['pred'][:,1].mean(axis=0):.3f}({dict_metric['pred'][:,1].std(axis=0):.3f})\n"
        f"Pred: FA: {dict_metric['pred'][:,2].mean(axis=0):.3f}({dict_metric['pred'][:,2].std(axis=0):.3f})\n"
    ))
    ##########################################################
    # export the dict of metrics as csv
    ##########################################################
    print('\nExporting metrics...')
    # Rank
    output_file_Rk = (
        f"{output_folder}"
        f"/[Rk]sim_nn_{args1.nn}"
        f"_pp_{args1.pp}"
        f"_qq_{args1.qq}"
        f"_rr_{args1.rr}"
        f"_stop_rr_{args1.stop_rr}"
        f"_const_qq_{args1.const_qq}"
        f"_rhoX_{args1.rhoX}"
        f"_bb_{args1.bb}"
        f"_nrep_{args1.nrep}"
        f"_ncores_{args1.ncores}"
        f"_subID_{args1.subID}"
        f"_kk_{args1.kk}"
        f"_const_num_itr_{args1.const_num_itr}"
        f"_gamma_{args1.gamma}.csv"
    )
    np.savetxt(output_file_Rk, dict_metric['rk'], delimiter=',')
    print('Rank saved!')
    # Est
    output_file_Est = (
        f"{output_folder}"
        f"/[Est]sim_nn_{args1.nn}"
        f"_pp_{args1.pp}"
        f"_qq_{args1.qq}"
        f"_rr_{args1.rr}"
        f"_stop_rr_{args1.stop_rr}"
        f"_const_qq_{args1.const_qq}"
        f"_rhoX_{args1.rhoX}"
        f"_bb_{args1.bb}"
        f"_nrep_{args1.nrep}"
        f"_ncores_{args1.ncores}"
        f"_subID_{args1.subID}"
        f"_kk_{args1.kk}"
        f"_const_num_itr_{args1.const_num_itr}"
        f"_gamma_{args1.gamma}.csv"
    )
    np.savetxt(output_file_Est, dict_metric['est'], delimiter=',')
    print('Est saved!')
    # Fit
    output_file_Fit = (
        f"{output_folder}"
        f"/[Fit]sim_nn_{args1.nn}"
        f"_pp_{args1.pp}"
        f"_qq_{args1.qq}"
        f"_rr_{args1.rr}"
        f"_stop_rr_{args1.stop_rr}"
        f"_const_qq_{args1.const_qq}"
        f"_rhoX_{args1.rhoX}"
        f"_bb_{args1.bb}"
        f"_nrep_{args1.nrep}"
        f"_ncores_{args1.ncores}"
        f"_subID_{args1.subID}"
        f"_kk_{args1.kk}"
        f"_const_num_itr_{args1.const_num_itr}"
        f"_gamma_{args1.gamma}.csv"
    )
    np.savetxt(output_file_Fit, dict_metric['fit'], delimiter=',')
    print('Fit saved!')
    # Pred
    output_file_Pred = (
        f"{output_folder}"
        f"/[Pred]sim_nn_{args1.nn}"
        f"_pp_{args1.pp}"
        f"_qq_{args1.qq}"
        f"_rr_{args1.rr}"
        f"_stop_rr_{args1.stop_rr}"
        f"_const_qq_{args1.const_qq}"
        f"_rhoX_{args1.rhoX}"
        f"_bb_{args1.bb}"
        f"_nrep_{args1.nrep}"
        f"_ncores_{args1.ncores}"
        f"_subID_{args1.subID}"
        f"_kk_{args1.kk}"
        f"_const_num_itr_{args1.const_num_itr}"
        f"_gamma_{args1.gamma}.csv"
    )
    np.savetxt(output_file_Pred, dict_metric['pred'], delimiter=',')
    print('Pred saved!')

if __name__ == "__main__":
    main()
