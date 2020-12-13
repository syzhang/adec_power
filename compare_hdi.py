"""
compare hdi from pystan fit traces
"""
import os, sys
import pickle
import numpy as np
import pandas as pd
from hbayesdm import rhat, print_fit, plot_hdi, hdi
import arviz as az
from matplotlib import pyplot as plt

def comp_hdi(model_name, param_ls, sort=True, draw_idx=50, draws=1000, seed=123):
    """
    compare hdi by drawing simulations
    """
    # define sim dir
    output_dir = './tmp_output/'+model_name+'_sim/'
    np.random.seed(seed)
    significant_df = []
    # find sim results in groups
    for key in param_ls:
        bounds = []
        c = 0
        # random compare n draws
        for comp in range(1,draws):
            # load MCMC traces with matching seeds (not number of subjects)
            hc_file = os.path.join(output_dir, 'hc_sim_'+str(int(np.random.randint(0,draw_idx,1)))+'.pkl')
            pt_file = os.path.join(output_dir, 'pt_sim_'+str(int(np.random.randint(0,draw_idx,1)))+'.pkl')

            if os.path.isfile(hc_file) and os.path.isfile(pt_file):
                # print(hc_file, pt_file)
                with open(hc_file, 'rb') as hc:
                    hc_dict = pickle.load(hc)
                with open(pt_file, 'rb') as pt:
                    pt_dict = pickle.load(pt)

                # calculate lower bounds of simulation using difference
                hdi_bounds = hdi_diff(key, hc_dict, pt_dict)
                # store hdi bounds
                bounds.append(hdi_bounds)
                c += 1
                # store sum dict
                if sort:
                    if comp == 1:
                        hc_sum = np.sort(hc_dict[key])
                        pt_sum = np.sort(pt_dict[key])
                        diff_sum = np.sort(hc_dict[key] - pt_dict[key])
                    else:
                        hc_sum += np.sort(hc_dict[key])
                        pt_sum += np.sort(pt_dict[key])
                        diff_sum += np.sort(hc_dict[key] - pt_dict[key])
                else:
                    if comp == 1:
                        hc_sum = hc_dict[key]
                        pt_sum = pt_dict[key]
                        diff_sum = hc_dict[key] - pt_dict[key]
                    else:
                        hc_sum += hc_dict[key]
                        pt_sum += pt_dict[key]
                        diff_sum += hc_dict[key] - pt_dict[key]
        # percentage of significant draws (ie bounds doesn't encompass 0)
        significant_pc = hdi_stats(key, bounds)
        significant_df.append(significant_pc)
        # plot hdi 
        plot_hdi_groups(model_name, key, hc_sum/float(draws), pt_sum/float(draws), sort)
        # plot hdi 
        plot_hdi_diff(model_name, key, diff_sum/float(draws), sort)
    # save significant calculation
    df_sig = pd.DataFrame({'parameter': param_ls,
                        'significant_percent': significant_df},index=None)
    df_sig.to_csv('./figs/'+model_name+'/significance_pc.csv')

def hdi_stats(key, hdi_bounds):
    """calculate hdi bounds stats"""
    # print(hdi_bounds)
    dfb = pd.DataFrame(hdi_bounds)
    dfb.columns = ['lower', 'upper']
    significant_sim = 0
    for idx,row in dfb.iterrows():
        if np.sign(row['lower']) == np.sign(row['upper']):
            significant_sim += 1
    significant_pc = significant_sim/dfb.shape[0]*100.
    print(f'{key} significant %: {significant_pc:.2f}')
    return significant_pc
        

def hdi_diff(param_key, hc_dict, pt_dict):
    """calculate difference between patient and control group mean param traces"""
    # calculate difference between groups
    param_diff = hc_dict[param_key] - pt_dict[param_key]
    # calculate hdi
    hdi_bounds = hdi(param_diff)
    # print(param_key+' hdi range: ', hdi_bounds)
    return hdi_bounds

def plot_hdi_groups(model_name, param_key, hc_dict, pt_dict, sort, credible_interval=0.94, point_estimate='mean', bins='auto', round_to=2):
    """plotting param posterior in groups"""
    # kwargs.setdefault('color', 'black')
    # x = [hc_dict[param_key], pt_dict[param_key]]
    x = [hc_dict, pt_dict]
    leg = [['Control'], ['Patient']]
    colors = ['blue','green']
    fig, axes = plt.subplots(2,1, sharex=True)
    for i in range(2):
        ax = axes[i]
        az.plot_posterior(x[i],
                        ax=ax,
                        kind='hist',
                        credible_interval=credible_interval,
                        point_estimate=point_estimate,
                        bins=bins,
                        round_to=round_to,
                        color=colors[i])
        if i ==0:
            title = param_key
        else:
            title = ''
        ax.set_title(title)
        ax.legend(leg[i])
    # save fig
    save_dir = './figs/'+model_name+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if sort:
        save_name = param_key+'_hdi_sorted.png' 
    else:
        save_name = param_key+'_hdi.png'
    fig = ax.get_figure()
    fig.savefig(save_dir+save_name)

def plot_hdi_diff(model_name, param_key, diff_dict, sort, credible_interval=0.94, point_estimate='mean', bins='auto', round_to=2):
    """plotting param posterior in diff"""
    # kwargs.setdefault('color', 'black')
    # x = [hc_dict[param_key], pt_dict[param_key]]
    x = diff_dict
    fig, ax = plt.subplots(1,1, sharex=True)
    az.plot_posterior(x,
                    ax=ax,
                    kind='hist',
                    credible_interval=credible_interval,
                    point_estimate=point_estimate,
                    bins=bins,
                    round_to=round_to,
                    color='black')
    title = param_key
    ax.set_title(title)
    ax.legend(['Control-Patient'])
    # save fig
    save_dir = './figs/'+model_name+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if sort:
        save_name = param_key+'_hdi_diff_sorted.png'
    else:
        save_name = param_key+'_hdi_diff.png'
    fig = ax.get_figure()
    fig.savefig(save_dir+save_name)

if __name__ == "__main__":
    # arg
    model_name = sys.argv[1] # which model sims to compare

    if model_name == 'bandit':
        # output_dir = './tmp_output/bandit_sim/'
        param_ls = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi']
    elif model_name == 'generalise':
        # output_dir = './tmp_output/generalise_sim/'
        param_ls = ['mu_sigma_a', 'mu_sigma_n', 'mu_eta', 'mu_kappa', 'mu_beta', 'mu_bias']
    else:
        print('model must be bandit or generalise.')
    comp_hdi(model_name, param_ls, sort=True, draw_idx=51, draws=1000)