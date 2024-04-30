#!/bin/bash
# NQS vs Grover

python [exp]nqs_grover.py --outDIR ./ --pp 8 --const_num_itr 3 --kk 9 --gamma 0.5 --ncores 10 --nrep 100  --seed1 5

python [exp]nqs_grover.py --outDIR ./ --pp 9 --const_num_itr 3 --kk 9 --gamma 0.5 --ncores 10 --nrep 100  --seed1 5

python [exp]nqs_grover.py --outDIR ./ --pp 10 --const_num_itr 3 --kk 9 --gamma 0.5 --ncores 10 --nrep 100  --seed1 5
