# README.md

## Folder py_packages contains the **qics** library which implements FA, NQS and RNQS. NQS is the core module of the proposed method QICS.
Use *python setup.py install* to install.


## Folder simulation includes the code for simulation studies in Section 5.1.
*STRS_OLS.sh* is the main bash script that produces the results of STRS and OLS.
*[exp]sim.sh* calls *[exp]sim.py* and generates the results for ES, QICS, and FA.
*[exp]pgd2.sh* calls *[exp]pgd2.py* that implements TR and generates the results.


## Folder lr contains the Python/R code and data to perform sensitivity analysis for leaning rate in Section 5.2.
*[expFD]lr.py* produces the prediction errors of QICS with varying learning rates.
*vis_lr.R* creates the plot of prediction error against learning rate.


## Folder nqs_grv provides the Python code for the NQS-Grover comparison in Section 5.3.
*[exp]nqs_grover.py* implements the comparisons between NQS and Grover's algorithm under various scenarios. *[exp]nqs_grover.sh* calls the python script to generate the results.


## Folder pm25 includes the code for the real data example in Appendix A.2.
*[expRD]pm25.sh* generates the results for ES, QICS, and FA based on *[expRD]pm25.py*.
*[expRD]pm25_pgd2.sh* calls *[expRD]pm25_pgd2.py* and generates the results for TR.
*[expRD]pm25.R* is an R script to produce the results of STRS and OLS.
*plotmap.R* creates the visual associations between the 4 pollutants and PM2.5.
