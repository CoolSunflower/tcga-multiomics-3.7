# TCGA Multiomics Feature Selection Study Comprehensive Codebase

This repository contains a comprehensive codebase for the execution of Machine Learning (ML) tasks considering various input criteria. 

Setup Guide: https://docs.google.com/document/d/12HQAhyIb7gWdJ9vKBG7I5ty9nf2XZrsJmMUo0PVpNOY/edit?usp=sharing

Please make sure to read the details on which input argument is required for which condition in `main.py` file.

## Global paths

Before running the scripts, ensure to verify the global paths of the dataset and the current folder in which you are working. In the Python code `main.py`, the default paths are defined as follows:

    folderISAAC = 'GenderBidirectionalTransfer/'
    if os.path.exists(folderISAAC)!=True:
        folderISAAC = './'

## Acknowledgement

This work has been supported by NIH R01 grant.


## Contact

For any queries, please contact:
```
Prof. Yan Cui (ycui2@uthsc.edu)
Dr. Teena Sharma (tee.shar6@gmail.com)
Adarsh Gupta (adarsh.gupta@iitg.ac.in/iamadarshgupta8@gmail.com)
```
