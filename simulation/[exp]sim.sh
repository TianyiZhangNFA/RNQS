#!/bin/bash


python [exp]sim.py --outDIR ./hd --nn 100 --pp 50 --qq 15 --rr 3 --stop_rr 5 --const_qq 23 --rhoX 0.1 --bb 0.4 --nrep 200 --ncores 20 --subID 5 --const_num_itr 3 --kk 5 --gamma 0.5

python [exp]sim.py --outDIR ./hd --nn 100 --pp 50 --qq 15 --rr 3 --stop_rr 5 --const_qq 23 --rhoX 0.1 --bb 0.7 --nrep 200 --ncores 20 --subID 5 --const_num_itr 3 --kk 5 --gamma 0.5

python [exp]sim.py --outDIR ./hd --nn 100 --pp 50 --qq 15 --rr 3 --stop_rr 5 --const_qq 23 --rhoX 0.5 --bb 0.4 --nrep 200 --ncores 20 --subID 5 --const_num_itr 3 --kk 5 --gamma 0.5

python [exp]sim.py --outDIR ./hd --nn 100 --pp 50 --qq 15 --rr 3 --stop_rr 5 --const_qq 23 --rhoX 0.5 --bb 0.7 --nrep 200 --ncores 20 --subID 5 --const_num_itr 3 --kk 5 --gamma 0.5
