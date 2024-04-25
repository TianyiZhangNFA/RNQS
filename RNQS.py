#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:47:33 2023

@author: Tianyi Zhang
"""

import numpy as np
import random
import math
from collections import Counter


  
#############################
# function: Grover Iteration
def grover_itr(loss_list, oracle_idx, ampl_list):
        """
        run Grover iteration once

        Input: 
            [list] loss_list
            [int] oracle_idx
            [list] ampl_list
        Return: 
            [list] amplified amplitudes
        """
        # loss value of the oracle state
        loss_oracle = loss_list[oracle_idx]
        # 1. flipped signs of amplitudes
        ampl_flipped = [-ampl_i if loss_i <=
                        loss_oracle else ampl_i
                        for loss_i, ampl_i in zip(loss_list, ampl_list)]
        # 2. inversion about the mean
        ampl_mean = sum(ampl_flipped)/len(ampl_flipped)
        ampl_inv = [-ampl_i + 2*ampl_mean for ampl_i in ampl_flipped]
        return ampl_inv

##################################################
# function: Grover's Search Algorithm
def groverSearch(n,loss_list, oracle_idx=None, seed=None):
        """
        Grover's search algorithm
        
        Input: 
            [list] loss_list
            [int] oracle_idx
            [int] seed
            [list] self.ST_SUPPOS
            [function] self.grover_itr
            [int] self.N

        Return: 
            [int] index of the minimal value in the loss_list
            [float] minimium of the loss_list
        """
        N=2**n
        ST_SUPPOS = [1./math.sqrt(N)]*N
        if seed is not None:
            # fix the random seed
            random.seed(seed)
        # initialize oracle_idx if not given
        if oracle_idx is None:
            oracle_idx = random.choice(range(N))
        # theoretical number of Grover iterations
        opt_itr = math.ceil( math.sqrt(N)*(math.pi/4) )
        # initialize the state as superposition
        phi_tmp = ST_SUPPOS.copy()
        # run Grover iteration itr times
        for _ in range(opt_itr):
            phi_tmp = grover_itr(loss_list, oracle_idx, phi_tmp)
        # measure the quantum register denoted by y?? the final state is generated randomly?Q2
        prob1 = [phi_i**2 for phi_i in phi_tmp]
        y = random.choices(range(N), weights=prob1, k=1)[0]
        return y, loss_list[y]

def NQS1(n, loss_list, num_itr, oracle_idx=None, seed=None, verbose=0, gamma=0.5):
        """
        Non-oracular Quantum Search (one chain only)
        
        Input: 
        External:
            [list] loss_list: a list of loss values
            [int] num_itr: number of iterations
            [int] oracle_idx: index of an initial oracle
            [int] seed: random seed
            [int] verbose: whether output the process with default 0
            [float] gamma: learning rate with default 0.5  

        Return: 
            [int] index of the minimal value in loss_list
            [float] minimium of loss_list
        """

        N=2**n
        ST_SUPPOS = [1./math.sqrt(N)]*N
        if seed is not None:
            # fix the random seed
            random.seed(seed)
        # initialize oracle_idx if not given
        if oracle_idx is None:
            oracle_idx = random.choice(range(N))
        if verbose:
            print(f"Initial oracle: {oracle_idx}")
        # main process
        for i_itr in range(1, num_itr+1):
            # number of Grover Iterations in this round
            itr = math.ceil(gamma**(-0.5*i_itr) * math.pi/4)
            # initialize the state as superposition ???Q3 in each interation, only start from equally random?
            phi_tmp = ST_SUPPOS.copy()
            # run Grover Iterations itr times
            for _ in range(itr):
                phi_tmp = grover_itr(loss_list, oracle_idx, phi_tmp)
            # take a measurement denoted by y
            prob1 = [phi_i**2 for phi_i in phi_tmp]
            y = random.choices(range(N),
                               weights=prob1,
                               k=1)[0]
            # update oracle_idx if y is better in terms of loss
            if loss_list[y] <= loss_list[oracle_idx]: 
                oracle_idx = y                 
            # print current oracle_idx
            if verbose:
                print(f"Round {i_itr+1}/{num_itr}: oracle is {oracle_idx}")

        return oracle_idx, loss_list[oracle_idx]
    
######################################################
# function: Non-oracular Quantum Search (multiple chains + majority voting)
def NQS(n, loss_list, num_itr, kk=1, oracle_idx=None, seed=None, verbose=0, gamma=0.5):
        """
        Non-oracular Quantum Search with multiple chains for majority voting
        
        Input: 
        External:
            [list] loss_list: a list of loss values
            [int] num_itr: number of iterations
            [int] kk: number of chains for majority voting
            [int] oracle_idx: index of an initial oracle
            [int] seed: random seed
            [int] verbose: whether output the process with default 0
            [float] gamma: learning rate with default 0.5  

        Return: 
            [int] index of the minimal value in loss_list
            [float] minimium of loss_list
        """

        # create a list of answers to vote
        qas_ans = []
        # generate a random seed if not given
        if seed is None:
            seed = random.randint(1, 1e4)
        # run multiple chains via a for-loop
        for seed_i in range(seed, seed+kk):
            min_idx_i, _ = NQS1(n,
                loss_list, num_itr, oracle_idx=oracle_idx, seed=seed_i, verbose=verbose, gamma=gamma)
            qas_ans.append(min_idx_i)
        # majority voting
        min_idx = Counter(qas_ans).most_common(1)[0][0]
        return min_idx, loss_list[min_idx]


def RNQS1(n, loss_list, num_itr, n_measures, oracle_idx=None, seed=None, verbose=0, gamma=0.5):
        """
        Non-oracular Quantum Search (one chain only)
        
        Input: 
        External:
            [list] loss_list: a list of loss values
            [int] num_itr: number of iterations
            [int] n_measures: number of minimum voting
            [int] oracle_idx: index of an initial oracle
            [int] seed: random seed
            [int] verbose: whether output the process with default 0
            [float] gamma: learning rate with default 0.5

        Return: 
            [int] index of the minimal value in loss_list
            [float] minimium of loss_list
        """
        N=2**n
        ST_SUPPOS = [1./math.sqrt(N)]*N
        if seed is not None:
            # fix the random seed
            random.seed(seed)
        # initialize oracle_idx if not given
        if oracle_idx is None:
            oracle_idx = random.choice(range(N))
        if verbose:
            print(f"Initial oracle: {oracle_idx}")
        # main process
        for i_itr in range(1, num_itr+1):
            # number of Grover Iterations in this round
            itr = math.ceil(gamma**(-0.5*i_itr) * math.pi/4)
            # initialize the state as superposition
            phi_tmp = ST_SUPPOS.copy()
            # run Grover Iterations itr times
            for _ in range(itr):
                phi_tmp = grover_itr(loss_list, oracle_idx, phi_tmp)
            # take a measurement denoted by y
            prob1 = [phi_i**2 for phi_i in phi_tmp]
            y_list = random.choices(range(N),
                                    weights=prob1,
                                    k=n_measures)
            y_loss=[loss_list[i] for i in y_list]
            y=y_list[np.argmin(y_loss)]
            # update oracle_idx if y is better in terms of loss
            if loss_list[y] <= loss_list[oracle_idx]: 
                oracle_idx = y                    
            # print current oracle_idx
            if verbose:
                print(f"Round {i_itr+1}/{num_itr}: oracle is {oracle_idx}")

        return oracle_idx, loss_list[oracle_idx]