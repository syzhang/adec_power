"""
compare hdi from pystan fit traces
"""
import os
import pickle
import numpy as np
from hbayesdm import rhat, print_fit, plot_hdi, hdi

output_dir = './tmp_output/'

param_ls = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi']

np.random.seed(123)
# find sim results in groups
for key in param_ls:
    upper, lower = [], []
    # random compare 1000 draws
    for comp in range(1,1000):
        # load MCMC traces with matching seeds (not number of subjects)
        hc_file = os.path.join(output_dir, 'hc_sim_'+str(int(np.random.randint(1,50,1)))+'.pkl')
        pt_file = os.path.join(output_dir, 'pt_sim_'+str(int(np.random.randint(1,50,1)))+'.pkl')

        if os.path.isfile(hc_file) and os.path.isfile(pt_file):
            # print(hc_file, pt_file)
            with open(hc_file, 'rb') as hc:
                hc_dict = pickle.load(hc)
            with open(pt_file, 'rb') as pt:
                pt_dict = pickle.load(pt)

            # calculate difference between groups
            param_diff = hc_dict[key] - pt_dict[key]
            # calculate hdi
            hdi_bounds = hdi(param_diff)
            # print(key+' hdi range: ', hdi_bounds)
            # store hdi bounds
            lower.append(hdi_bounds[0])
            upper.append(hdi_bounds[1])
    # percentage of lower bounds above 0 (ie bounds doesn't encompass 0)
    print(key+' above zero: ', sum(np.array(lower)>0)/len(lower))
