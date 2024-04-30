import random
import math
from collections import Counter

######################################################
# Class: Quantum Search and Forward Adding
class qSearch():
    #############################
    # Initialization
    def __init__(self, N):
        self.N = N
        self.ST_SUPPOS = [1./math.sqrt(self.N)]*self.N
    
    #############################
    # function: Grover Iteration
    def grover_itr(self, loss_list, oracle_idx, ampl_list):
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
        # 2. inversion about the mean ????Q1 this step should inverse with psy_0?
        ampl_mean = sum(ampl_flipped)/len(ampl_flipped)
        ampl_inv = [-ampl_i + 2*ampl_mean for ampl_i in ampl_flipped]
        return ampl_inv

    ##################################################
    # function: Grover's Search Algorithm
    def groverSearch(self, loss_list, oracle_idx=None, seed=None):
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
        if seed is not None:
            # fix the random seed
            random.seed(seed)
        # initialize oracle_idx if not given
        if oracle_idx is None:
            oracle_idx = random.choice(range(self.N))
        # theoretical number of Grover iterations
        opt_itr = math.ceil( math.sqrt(self.N)*(math.pi/4) )
        # initialize the state as superposition
        phi_tmp = self.ST_SUPPOS.copy()
        # run Grover iteration itr times
        for _ in range(opt_itr):
            phi_tmp = self.grover_itr(loss_list, oracle_idx, phi_tmp)
        # measure the quantum register denoted by y?? the final state is generated randomly?Q2
        prob1 = [phi_i**2 for phi_i in phi_tmp]
        y = random.choices(range(self.N), weights=prob1, k=1)[0]
        return y, loss_list[y]
    
    #####################################
    # function: Non-oracular Quantum Search (one chain)
    def NQS1(self, loss_list, num_itr, oracle_idx=None, seed=None, verbose=0, gamma=0.5):
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
        Internal:
            [list] self.ST_SUPPOS: a list of equally-weighted amplititudes
            [function] self.grover_itr: a function to run one single Grover Iteration
            [int] self.N: number of elements in loss_list

        Return: 
            [int] index of the minimal value in loss_list
            [float] minimium of loss_list
        """
        if seed is not None:
            # fix the random seed
            random.seed(seed)
        # initialize oracle_idx if not given
        if oracle_idx is None:
            oracle_idx = random.choice(range(self.N))
        if verbose:
            print(f"Initial oracle: {oracle_idx}")
        # main process
        for i_itr in range(1, num_itr+1):
            # number of Grover Iterations in this round
            itr = math.ceil(gamma**(-0.5*i_itr) * math.pi/4)
            # initialize the state as superposition ???Q3 in each interation, only start from equally random?
            phi_tmp = self.ST_SUPPOS.copy()
            # run Grover Iterations itr times
            for _ in range(itr):
                phi_tmp = self.grover_itr(loss_list, oracle_idx, phi_tmp)
            # take a measurement denoted by y
            prob1 = [phi_i**2 for phi_i in phi_tmp]
            y = random.choices(range(self.N),
                               weights=prob1,
                               k=1)[0]
            # update oracle_idx if y is better in terms of loss
            if loss_list[y] <= loss_list[oracle_idx]: ##Q4 we already know the lost list, NQS must be faster than the seach but the advance is that NQS do not need calculate all loss
                oracle_idx = y                        ## but we do not need to search all results and it is not NP hard
                ##every step we start from equally random then tends to the target gradually, if in the process the loss decreases then renew the target.
                ## This is not related to the lr problem, it is finding minimum
            # print current oracle_idx
            if verbose:
                print(f"Round {i_itr+1}/{num_itr}: oracle is {oracle_idx}")

        return oracle_idx, loss_list[oracle_idx]

    ######################################################
    # function: Non-oracular Quantum Search (multiple chains + majority voting)
    def NQS(self, loss_list, num_itr, kk=1, oracle_idx=None, seed=None, verbose=0, gamma=0.5):
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
            min_idx_i, _ = self.NQS1(
                loss_list, num_itr, oracle_idx=oracle_idx, seed=seed_i, verbose=verbose, gamma=gamma)
            qas_ans.append(min_idx_i)
        # majority voting  ?? Q5 Why not find the one with the smallest loss
        min_idx = Counter(qas_ans).most_common(1)[0][0]
        return min_idx, loss_list[min_idx]

    ######################################################
    # Function: perform forward adding to find the minimum
    def FA(self, loss_list, comb_list, stop_rr):
        """
        perform forward adding to find the minimum

        Input:
            [list]loss_list: list to find the minimum from
            [list]comb_list: a list of tuples, combinations of support of matrix A1
            [int]stop_rr: upper limit of rank

        Return: 
            [int] index of the minimum in loss_list
            [float] minimium of loss_list
            [tuple] corresponding tuple of the minimum
        """

        # initialization
        idx_start = 0
        best_sup = (0,)
        best_val = max(loss_list)
        best_idx = 0

        for icomb in range(1, stop_rr+1):
            # calculate number of candidate models
            # math.comb is new in python 3.8
            tmp_nums = math.comb(int(math.log2(self.N)), icomb)
            # select candidate models via for-loop
            for tmp_idx in range(idx_start, idx_start+tmp_nums):
                # support and value for current model
                tmp_sup, tmp_val = comb_list[tmp_idx], loss_list[tmp_idx]
                # pass if first-time search
                if idx_start == 0:
                    pass
                # skip the following if current set is not a subset of upcoming candidate
                elif (set(best_sup).issubset(set(tmp_sup))) == False:
                    continue
                # update support/index if current is smaller
                if best_val >= tmp_val:
                    best_sup, best_idx, best_val = tmp_sup, tmp_idx, tmp_val 
            idx_start += tmp_nums

        return best_idx, best_val, best_sup
######################################################
# Function: pow2list ceils a list to one of length power of 2


def pow2list(loss_list, powOutput):
    """
    pow2list: augment loss_list of length power of 2

    Input:
        [list]loss_list
        [int]powOutput: 2 to the power of listOutput

    Return:
        [list]listOutput: augmented list which contains loss_list followed by knockoffs (copies of maximum value in loss_list)
    """
    lenInput, maxInput = len(loss_list), max(loss_list)
    lenKnockoff = 2**powOutput - lenInput
    listKnockoff = [maxInput]*lenKnockoff
    listOutput = loss_list + listKnockoff
    return listOutput
