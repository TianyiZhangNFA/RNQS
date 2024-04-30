from qics.QICS_func import qSearch, pow2list  # random, math, Counter included
import math
import numpy as np
import argparse
import time
import multiprocessing
from functools import partial
import pickle
from operator import itemgetter # index-based item extraction from an iterable where each element is a tuple of multiple elements
import re
import sys
from pathlib import Path
######################################################
# read arguments
######################################################

parser = argparse.ArgumentParser()
parser.add_argument('--inFile',
                    default='.pkl',
                    type=str,
                    metavar='',
                    help='input file')
parser.add_argument('--outDIR',
                    default='.',
                    type=str,
                    metavar='',
                    help='output directory')
parser.add_argument('--const_num_itr',
                    default=3,
                    type=int,
                    metavar='',
                    help='multiplier for calculating number of iterations in NQS')
parser.add_argument('--kk',
                    default=5,
                    type=int,
                    metavar='',
                    help='number of chains in NQS')
parser.add_argument('--gamma',
                    default=0.5,
                    type=float,
                    metavar='',
                    help='learning rate in NQS')
parser.add_argument('--ncores',
                    default=10,
                    type=int,
                    metavar='',
                    help='number of cores')
args1 = parser.parse_args()

####################################################################


class argsDigest():
    """
    automatically extract parameters of the experiment from the file name
    """
    def __init__(self, inFile):
        ############################################################
        match_qq = re.match(r"^.*qq_([0-9]{0,3})_.*$", inFile)
        if match_qq is None:
            print('qq is NOT found! Please check the file name.')
            sys.exit()
        else:
            self.qq = int(match_qq.group(1))
        ############################################################
        match_rr = re.match(r"^.*rr_([0-9]{0,3})_.*$", inFile)
        if match_rr is None:
            print('rr is NOT found! Please check the file name.')
            sys.exit()
        else:
            self.rr = int(match_rr.group(1))
        ############################################################
        match_stop_rr = re.match(r"^.*stop_rr_([0-9]{0,3})_.*$", inFile)
        if match_stop_rr is None:
            print('stop_rr is NOT found! Please check the file name.')
            sys.exit()
        else:
            self.stop_rr = int(match_stop_rr.group(1))
        ############################################################
        match_rhoX = re.match(r"^.*rhoX_([0-9.]{0,3})_.*$", inFile)
        if match_stop_rr is None:
            print('rhoX is NOT found! Please check the file name.')
            sys.exit()
        else:
            self.rhoX = float(match_rhoX.group(1))
        ############################################################
        match_bb = re.match(r"^.*bb_([0-9.]{0,3})_.*$", inFile)
        if match_bb is None:
            print('bb is NOT found! Please check the file name.')
            sys.exit()
        else:
            self.bb = float(match_bb.group(1))
        ############################################################
        match_nrep = re.match(r"^.*nrep_([0-9]{0,5})_.*$", inFile)
        if match_nrep is None:
            print('nrep is NOT found! Please check the file name.')
            sys.exit()
        else:
            self.nrep = int(match_nrep.group(1))
        ############################################################
        self.all = [self.qq,
                    self.rr,
                    self.stop_rr,
                    self.rhoX,
                    self.bb,
                    self.nrep]
    def __len__(self):
        return len(self.all)
    def __getitem__(self, index):
        return self.all[index]

####################################################################
# %%

def worker_func(args1, args2, qSearch, tuples_ALL, supList, oracle_idx, i_rep):
    """find the minimum in parallel

    Args:
        args1 (class): input parameters
        args2 (class): extracted parameters of experiment setting from the input file name
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
    idxQICS, _ = qsearch.qAdaSearch(LOSS_LIST_AG, num_itr, kk=args1.kk, oracle_idx=oracle_idx, seed=seed, gamma=args1.gamma)
    return idxES, idxQICS


######################################################
# main
######################################################


def main():
    ##########################################################
    ## pickle loading
    print(f"Loading {args1.inFile}...")
    with open(args1.inFile, 'rb') as f:
        tuples_ALL = pickle.load(f)
    print('Data loaded!')
    ##########################################################
    ## parameter extraction
    args2 = argsDigest(inFile=args1.inFile)
    print('Parameters extracted!')
    # print(args2.all)
    ##########################################################
    output_folder = f"{args1.outDIR}/r{args2.rhoX}b{args2.bb}"
    Path(output_folder).mkdir(exist_ok=True)
    print('Folder %s created!' % output_folder)
    output_folder = f"{output_folder}/k{args1.kk}"
    Path(output_folder).mkdir(exist_ok=True)
    print('Folder %s created!' % output_folder)
    ##########################################################
    ## i[7]: Support of idx_A1
    ## i[8]: Each element in iterable repeat nrep times: e.g.
    ## [0,1,2,...,nrep-1,0,1,...,nrep-1,...,nrep-1]
    supList = [i[7] for i in tuples_ALL if i[8] == 0]
    # iterable for multiprocessing
    rep_loop = range(args2.nrep)
    ##########################################################
    ## access metric values using the indices in a tuple via itemgetter
    ## 0: rank estimate (Rank)
    ## 3: scaled estimation error (Est)
    ## 4: scaled RMSE (Fit)
    ## 5: scaled validation RMSE (Fit)
    ## 6: scaled prediction RMSE (Pred)
    getter1 = itemgetter(0,5)
    ##########################################################
    # evaluate ES/QICS
    # oracle index for QICS
    oracle_idx = None
    # pool object
    with multiprocessing.Pool(args1.ncores) as mp:
        # reduce the numper of arugments via partial function
        part_func = partial(worker_func, args1, args2, qSearch,
                            tuples_ALL, supList, oracle_idx)
        # experiment starts
        starttime = time.time()
        print('Start!')
        res_tuples = mp.map(part_func, rep_loop)
        # experiment ends
        print("DONE!")
    print(f"That took {time.time() - starttime} seconds\n")
    ##########################################################
    # extract keywords from the source file
    match1 = re.match(
        r"^/home.*(sim_.*)_ncores.*_(subID_.*).pkl$", args1.inFile)
    if match1 is None:
        print('Format of the input file is invalid!')
    # export a list of tuples as pickle
    output_file = (
        f"{output_folder}/[LR]{match1.group(1)}"
        f"_{match1.group(2)}"
        f"_kk_{args1.kk}"
        f"_const_num_itr_{args1.const_num_itr}"
        f"_gamma_{args1.gamma}"
        f"_ncores_{args1.ncores}.pkl"
    )
    with open(output_file, 'wb') as f:
        pickle.dump(res_tuples, f)
    print('Data saved!\n')
    ##########################################################
    # calculate measures
    ##########################################################
    # calculate coverage percentage
    qAda_pct = [1 if idx_i == truth_i else 0 for truth_i,
                idx_i in res_tuples]
    print(f"QICS Accuracy Rate: {np.mean(qAda_pct):.2f}\n")
    # create a dict of metrics
    dict_metric = {'rk': np.zeros((args2.nrep, 2)),
                    'val': np.zeros((args2.nrep, 2))
                    }
    for i, (iES, iQICS) in enumerate(res_tuples):
        # ES
        dict_metric['rk'][i, 0], dict_metric['val'][i, 0] = getter1(tuples_ALL[i+iES*args2.nrep])
        # QICS
        dict_metric['rk'][i, 1], dict_metric['val'][i, 1] = getter1(tuples_ALL[i+iQICS*args2.nrep])
    print((
        f"Criterion: ES,QICS mean(std)\n"
        # Rank
        f"Rank: ES: {dict_metric['rk'][:,0].mean(axis=0):.3f}({dict_metric['rk'][:,0].std(axis=0):.3f})\n"
        f"Rank: QICS: {dict_metric['rk'][:,1].mean(axis=0):.3f}({dict_metric['rk'][:,1].std(axis=0):.3f})\n"
        # Val
        f"Val: ES: {dict_metric['val'][:,0].mean(axis=0):.3f}({dict_metric['val'][:,0].std(axis=0):.3f})\n"
        f"Val: QICS: {dict_metric['val'][:,1].mean(axis=0):.3f}({dict_metric['val'][:,1].std(axis=0):.3f})\n"
    ))
    

if __name__ == "__main__":
    main()
