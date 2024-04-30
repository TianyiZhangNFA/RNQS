import sys

sys.path.append('/path/to/application/app/folder')
from py_packages.qics.QICS_func import qSearch  # random, math, Counter included
import time
import random
import math
import multiprocessing
import argparse
from functools import partial
import pickle
import numpy as np
from pathlib import Path
######################################################
# read arguments
######################################################

parser = argparse.ArgumentParser()
parser.add_argument('--outDIR',
                    default='./',
                    type=str,
                    metavar='',
                    help='output directory')
parser.add_argument('--pp',
                    default=8,
                    type=int,
                    metavar='',
                    help='number of predictors')
parser.add_argument('--const_num_itr',
                    default=3,
                    type=int,
                    metavar='',
                    help='multiplier for calculating number of iterations in NQS')
parser.add_argument('--kk',
                    default=9,
                    type=int,
                    metavar='',
                    help='number of chains in NQS')
parser.add_argument('--gamma',
                    default=0.5,
                    type=float,
                    metavar='',
                    help='learning rate in NQS')
parser.add_argument('--seed1',
                    default=2,
                    type=int,
                    metavar='',
                    help='seed for NQS and Grover')
parser.add_argument('--ncores',
                    default=10,
                    type=int,
                    metavar='',
                    help='number of cores')
parser.add_argument('--nrep',
                    default=100,
                    type=int,
                    metavar='',
                    help='number of replicates')
args1 = parser.parse_args()


def worker_func(args1, qSearch, i_rep):
    """find the minimum in parallel

    Args:
        args1 (class): input parameters
        qSearch (class): quantum search module
        i_rep (int): ith-iteration

    Returns:
        sum(sol_NQS)/100 (int): coverage percentage by NQS
        sum(sol_Grv)/100 (int): coverage percentage by Grover
    """

    ##################################
    # Data Generating Process
    ##################################
    # a list of U[0,1] of size 2**p
    random.seed(91*i_rep+44) # seed
    data_list = [random.random() for _ in range(2**args1.pp)]
    # set number of iterations for NQS
    num_itr = math.ceil(args1.const_num_itr*2*math.log2(args1.pp))
    qsearch = qSearch(2**args1.pp)
    
    ##################################
    # Execution
    ##################################
    # Exhaustive Search(ES)
    idxGT = data_list.index(min(data_list))
    # result lists of NQS and grover
    sol_NQS, sol_Grv = [0]*100, [0]*100
    # run 100 replicates
    for ii_rep in range(100):
        # seed
        seed_ii = 17*ii_rep+args1.seed1
        # NQS
        idxNQS, _ = qsearch.qAdaSearch(
            data_list, num_itr, kk=args1.kk, oracle_idx=None, 
            seed=seed_ii, gamma=args1.gamma)
        # Grover Search 
        idxGrv, _ = qsearch.groverSearch(data_list, None, seed_ii)
        # save results
        if idxNQS == idxGT:
            sol_NQS[ii_rep] = 1
        if idxGrv == idxGT:
            sol_Grv[ii_rep] = 1

    return sum(sol_NQS)/100, sum(sol_Grv)/100

######################################################
# main
######################################################


def main():
    # iterable
    rep_loop = range(args1.nrep)
    # pool object
    with multiprocessing.Pool(args1.ncores) as mp:
        # reduce the numper of arugments via partial function
        part_func = partial(worker_func, args1, qSearch)
        # experiment starts
        starttime = time.time()
        print('Start!')
        res_tuples = mp.map(part_func, rep_loop)
        # experiment ends
        print("DONE!")

    
    print(f"That took {time.time() - starttime} seconds")

    Path(args1.outDIR).mkdir(exist_ok=True)
    print('Folder %s created!' % args1.outDIR)

    output_file = (
        f"{args1.outDIR}/[NQS_grover]{args1.pp}"
        f"_kk_{args1.kk}"
        f"_const_num_itr_{args1.const_num_itr}"
        f"_gamma_{args1.gamma}"
        f"_seed1_{args1.seed1}"
        f"_nrep_{args1.nrep}"
        f"_ncores_{args1.ncores}.pkl"
    )
    with open(output_file, 'wb') as f:
        pickle.dump(res_tuples, f)
    print('Data saved!\n')

    list_nqs, list_grv = [], []
    for i_nqs, i_grv in res_tuples:
        list_nqs.append(i_nqs)
        list_grv.append(i_grv)
    
    print( (
        f"Coverage_pct: mean (std)\n"
        f"Grover: {np.mean(list_grv):.3f}({np.std(list_grv):.3f})\n"
        f"NQS: {np.mean(list_nqs):.3f}({np.std(list_nqs):.3f})"
        ) )
                


if __name__ == "__main__":
    main()
