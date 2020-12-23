"""
simulated power calculation for temporal statistical learning task (ideal observer jump frequency model)
"""
import os, sys
import pickle
import numpy as np
import pandas as pd

def model_tsl_jumpfreq(param_dict, subjID, num_trial=200):
    """simulate temporal statistical learning task"""
    # generate bernoulli sequence with jumps
    p = 0.25
    pA = [p, 1-p, p, 1-p, p]
    L = np.ones(len(pA)) * num_trial # length of jump block
    N = sum(L)
    chklm = np.cumsum(L)
    pJump = (len(pA)-1)/sum(L)