"""
compare hdi from pystan fit traces
"""

import pickle
import numpy as np
from hbayesdm import rhat, print_fit, plot_hdi, hdi

name_str = '_sim_seed_1.pkl'

with open('./tmp_output/'+'hc'+name_str, 'rb') as ip:
    hc_dict = pickle.load(ip)
with open('./tmp_output/'+'pt'+name_str, 'rb') as ip:
    pt_dict = pickle.load(ip)

for key in hc_dict:
    if key.startswith('mu_'):
        # print(key)
        param_diff = hc_dict[key] - pt_dict[key]
        print(key+' hdi range: ', hdi(param_diff))
