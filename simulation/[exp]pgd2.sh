#!/bin/bash

##################################
# rhoX=0.1,0.5
# b=0.4,0.7


echo "p50q15r3rX.1b.4"
python /home/jc87685/pywork/qmr1221/py_scripts/[exp]pgd2.py --outDIR /home/jc87685/pywork/qmr1221/pgd_trace2 --nn 100 --pp 50 --qq 15 --rr 3 --rhoX 0.1 --bb 0.4 --nrep 200
date

echo "p50q15r3rX.1b.7"
python /home/jc87685/pywork/qmr1221/py_scripts/[exp]pgd2.py --outDIR /home/jc87685/pywork/qmr1221/pgd_trace2 --nn 100 --pp 50 --qq 15 --rr 3 --rhoX 0.1 --bb 0.7 --nrep 200
date

echo "p50q15r3rX.5b.4"
python /home/jc87685/pywork/qmr1221/py_scripts/[exp]pgd2.py --outDIR /home/jc87685/pywork/qmr1221/pgd_trace2 --nn 100 --pp 50 --qq 15 --rr 3 --rhoX 0.5 --bb 0.4 --nrep 200
date

echo "p50q15r3rX.5b.7"
python /home/jc87685/pywork/qmr1221/py_scripts/[exp]pgd2.py --outDIR /home/jc87685/pywork/qmr1221/pgd_trace2 --nn 100 --pp 50 --qq 15 --rr 3 --rhoX 0.5 --bb 0.7 --nrep 200
date
