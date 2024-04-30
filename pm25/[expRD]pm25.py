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
from itertools import combinations, chain
######################################################
# load algorithmic modules
######################################################
import math
import numpy as np
from qics.AEGD_func import AEGD, obj_val, norm_scaled #, loss2obj  # numpy included
from qics.QICS_func import qSearch, pow2list  # random, math, Counter included


"""
This script implements ES, QICS, FA on the pm2.5 real data example. 
"""


######################################################
# read arguments
######################################################

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
parser.add_argument('--stop_rr',
                    default=15,
                    type=int,
                    metavar='',
                    help='upper limit of rank search')
parser.add_argument('--const_qq',
                    default=20,
                    type=int,
                    metavar='',
                    help='a qq-related multiplier in penalty function')
parser.add_argument('--ncores',
                    default=5,
                    type=int,
                    metavar='',
                    help='number of cores')
parser.add_argument('--subID',
                    default=1,
                    type=int,
                    metavar='',
                    help='ID for replication purpose')
##################################################
# parameters for QICS
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
    i_itr: ith-iterable, idx_A1, e.g. (0,4)

    Return: 
    rr_est: rank estimate
    obj_est: obj value
    loss_est: loss value
    val_err: RMSE on val set
    test_err: RMSE on test set
    idx_A1: support of A1
    """

    idx_A1 = i_itr
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
    # rank estimate
    rr_est = len(idx_A1)
    qq = Ydata.shape[1]
    #########################################
    # OLS when A1 = A
    # do GD estimation as usual otherwise
    #########################################
    if rr_est == qq:
        # OLS estimation
        A_est, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)
    else:
        # all columns in matrix A
        idx_A = range(qq)
        idx_A2 = list(set(idx_A) - set(idx_A1))
        # generate Y1, Y2
        Y1 = Y_train[:, idx_A1]
        Y2 = Y_train[:, idx_A2]
        # learning rate
        lr = 10.**-(np.log10(n_trn) + 2)
        ##################################
        # AEGD-Estimation
        ##################################
        est_AEGD = AEGD()
        _, _, _, A_est = est_AEGD.est_GD(
            X_train, Y1, Y2, lr, 1e-6, 100)
    ##################################
    # Calculate objective value and loss value
    obj_est, loss_est, _ = obj_val(
        Y_train - X_train@A_est, rr_est, args1.const_qq)
    ##################################
    # Calculate error-based measures
    test_err = norm_scaled(X_test@A_est, Y_test)
    return rr_est, obj_est, loss_est, test_err, idx_A1

####################################################################
# worker function for Minimal Finding(MF)
def worker_func_MF(args1, qSearch, tuples_ALL, supList, oracle_idx):
    """find the minimum in parallel

    Args:
        args1 (class): input parameters
        qSearch (class): quantum search module
        tuples_ALL (tuple): a BIG tuple that contains all calculated values
        supList (list): a list of support of A1
        oracle_idx (int): the index of the minimal

    Returns:
        idxES(int): true indices of the minimal
        idxQICS(int): indices of the minimal by QICS
        idxFA (int): indices of the minimal by FA
    """
    ##################################
    # tuple_ALL explained
    ##################################
    # example/structure of an element of tuple_ALL
    # (1,                  [0] rr_est: rank/support size of A1 (Rank)
    # 2085.068328245891,   [1] obj_est: objective value to to tune FA
    # 2066.1367596765667,  [2] loss_est: raw loss value
    # 1.3650893500891532,  [3] test_err: scaled prediction RMSE (Pred)
    # (0,),                [4] idx_A1: support of A1
    # rr_est, obj_est, loss_est, test_err, idx_A1

    ##################################
    # Process the iterable (i_rep)
    ##################################
    # find the test error list for QICS and ES
    pred_list = [i[3] for i in tuples_ALL]
    lenPredList = len(pred_list)
    # find objective value list for FA
    obj_list = [i[1] for i in tuples_ALL]
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
    seed = 17*args1.subID+3
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
    print('\n===>Loss Generating')
    ##################################################
    # 2020.5.21 iterable using tuple (memory cost: tuple<list<set)
    itrs = (combinations(range(15), i) for i in range(1, args1.stop_rr+1))
    itr_chain = chain.from_iterable(itrs)
    ##################################################
    # pool object
    with multiprocessing.Pool(args1.ncores) as mp:
        # reduce the number of arugments via partial function
        part_func_DGP = partial(worker_func_DGP, args1)
        # experiment starts
        starttime = time.time()
        print('Start!')
        res_tuples_DGP = mp.map(part_func_DGP, itr_chain)
        # experiment ends
        print("DONE!")
    print(f"That took {time.time() - starttime} seconds")
    ##################################################
    # export DGP values as pickle
    output_file_DGP = (
        f"{args1.outDIR}"
        f"/[AEGD]pm25"
        f"_stop_rr_{args1.stop_rr}"
        f"_const_qq_{args1.const_qq}"
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
    ## i[5]: Support of idx_A1
    supList = [i[5] for i in res_tuples_DGP]
    
    ##########################################################
    ## access metric values using the indices in a tuple via itemgetter
    ## 0: rank estimate (Rank)
    ## 3: scaled prediction RMSE (Pred)
    getter1 = itemgetter(0, 3)
    ##########################################################
    # evaluate ES/QICS/FA
    # oracle index for QICS
    oracle_idx = None
    # experiment starts
    starttime = time.time()
    print('Start!')
    res_tuples_MF = worker_func_MF(args1, qSearch,
                                   res_tuples_DGP, supList, oracle_idx)
    # experiment ends
    print("DONE!")
    print(f"That took {time.time() - starttime} seconds\n")
    ##########################################################
    # export minimal findings as pickle
    output_file_MF = (
        f"{args1.outDIR}"
        f"/[MF]pm25"
        f"_stop_rr_{args1.stop_rr}"
        f"_const_qq_{args1.const_qq}"
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
    # calculate coverage 
    print(f"Quantum finds the true minimum: {res_tuples_MF[0] == res_tuples_MF[1]}\n")
    # create a dict of metrics
    dm = {'rk': np.zeros(3),
                   'pred': np.zeros(3)
                   }
    iES, iQICS, iFA = res_tuples_MF
    # ES
    dm['rk'][0], dm['pred'][0] = getter1(res_tuples_DGP[iES])
    # QICS
    dm['rk'][1], dm['pred'][1] = getter1(res_tuples_DGP[iQICS])
    # FA
    dm['rk'][2], dm['pred'][2] = getter1(res_tuples_DGP[iFA])
    print((
        f"Criterion: ES,QICS,FA\n"
        # Rank
        f"Rank: ES: {dm['rk'][0]:.3f}\n"
        f"Rank: QICS: {dm['rk'][1]:.3f}\n"
        f"Rank: FA: {dm['rk'][2]:.3f}\n"
        # Pred
        f"Pred: ES: {dm['pred'][0]:.3f}\n"
        f"Pred: QICS: {dm['pred'][1]:.3f}\n"
        f"Pred: FA: {dm['pred'][2]:.3f}\n"
    ))
    

if __name__ == "__main__":
    main()
