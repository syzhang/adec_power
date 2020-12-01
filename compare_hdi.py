"""
compare hdi from pystan fit traces
"""
import os
import pickle
import numpy as np
from hbayesdm import rhat, print_fit, plot_hdi, hdi

output_dir = './tmp_output/'

param_ls = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi']

# find sim results in groups
for key in param_ls:
    upper, lower = [], []
    for f in os.listdir(output_dir):
        if f.startswith('hc') and f.endswith('.pkl'):
            base_name = '_sim_' + f.split('_')[-1]
            hc_file = os.path.join(output_dir, 'hc'+base_name)
            pt_file = os.path.join(output_dir, 'pt'+base_name)
            if os.path.isfile(hc_file) and os.path.isfile(pt_file):
                # print(hc_file)
                with open(hc_file, 'rb') as hc:
                    hc_dict = pickle.load(hc)
                with open(pt_file, 'rb') as pt:
                    pt_dict = pickle.load(pt)

                # store hdi bounds
                # for key in hc_dict:
                #     if key.startswith('mu_'):
                        # print(key)
                param_diff = hc_dict[key] - pt_dict[key]
                hdi_bounds = hdi(param_diff)
                # print(key+' hdi range: ', hdi_bounds)
                lower.append(hdi_bounds[0])
                upper.append(hdi_bounds[1])
    print(key+' above zero: ', sum(np.array(lower)>0)/len(lower))
