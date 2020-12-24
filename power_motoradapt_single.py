"""
simulated power calculation for motor adaptation task (state space model)
"""
import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def sim_motoradapt_single(param_dict, sd_dict, group_name, seed, num_sj=50, num_trial=200, model_name='motoradapt_single', plot=False):
    """simulate with state space model for multiple subjects"""
    multi_subject = []
    
    # generate new params
    np.random.seed(seed)
    for sj in range(num_sj):
        sample_params = dict()
        for key in param_dict:
            sample_params[key] = np.random.normal(param_dict[key], sd_dict[key], size=1)[0]
        # print(sample_params)
        df_sj = model_motoradapt_single(sample_params, sj, num_trial)
        multi_subject.append(df_sj)
        
    df_out = pd.concat(multi_subject)
    # plot check
    if plot:
        plot_state(df_out)
    # saving output
    output_dir = './tmp_output/motoradapt_sim/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f_name = model_name+'_'+group_name+'_'+str(seed)
    df_out.to_csv(output_dir+f_name+'.txt', sep='\t', index=False)
    # print(df_out)

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def model_motoradapt_single(param_dict, subjID, num_trial=200):
    """state space model, single process model"""
    # rotation schedule (50 baseline trials, then rotation trials, 50 washout trials)
    rotation = np.concatenate([np.zeros(50), 45*np.ones(num_trial-100), np.zeros(50)])
    # rotation = np.concatenate([np.zeros(50), np.arange(0.,45.,45./50), 45.*np.ones(num_trial-150), np.zeros(50)])

    # initialise
    state_single = 0.
    sim_error = 0.
    # transform params
    A_retention = sigmoid(param_dict['A_retention'])
    B_learning = sigmoid(param_dict['B_learning'])
    # initialise output
    data_out = []
    # simulate trials
    for t in range(num_trial):
        # trial error, error = current state - perturbation
        sim_error = state_single - rotation[t]
        # output (add noise)
        single_state_rand = state_single + np.random.rand(1)[0]*10e-1
        data_out.append([subjID, t, single_state_rand, sim_error])
        # update, retained state - learning rate * error
        state_single = A_retention*state_single - B_learning*sim_error

    df_out = pd.DataFrame(data_out)
    df_out.columns = ['subjID', 'trial', 'state', 'error']

    return df_out

def plot_state(df_out):
    """plot state from model"""
    # df = df_out[df_out['subjID']==0]
    df = df_out
    fig = plt.subplots(figsize=(6,5))
    # plt.plot(df['state'], label='State')
    sns.lineplot(x='trial', y='state', data=df)
    plt.vlines(50, -10, 60, colors='black', linestyles='--')
    plt.vlines(max(df['trial'])-50, -10, 60, colors='black', linestyles='--')
    plt.hlines(0, 0, max(df['trial']), colors='black')
    plt.legend()
    plt.xlabel('Trial')
    plt.ylabel('Simluated reaching direction')
    plt.title('Single state-space model')
    plt.show()

if __name__ == "__main__":
    # healthy control parameters (made up based on Takiyama 2016)
    param_dict_hc = {
        'A_retention': 2.3,  # retention rate 0.910
        'B_learning': -0.6  # learning rate 0.344
    }
    # patient parameters (made up based on Takiyama 2016)
    param_dict_pt = {
        'A_retention': 2.5,  # retention rate 0.93
        'B_learning': -0.3  # learning rate 0.405
    }
    # healthy control sd
    sd_dict_hc = {
        'A_retention': 5,  # retention rate
        'B_learning': 2  # learning rate
    }
    # patient sd
    sd_dict_pt = {
        'A_retention': 5,  # retention rate
        'B_learning': 2  # learning rate
    }
    
    # parsing cl arguments
    group_name = sys.argv[1] # pt=patient, hc=control
    seed_num = int(sys.argv[2]) # seed number
    subj_num = int(sys.argv[3]) # subject number to simulate
    trial_num = int(sys.argv[4]) # trial number to simulate

    # simulate
    model_name = 'motoradapt_single'
    if group_name == 'hc':
        # simulate hc subjects with given params
        sim_motoradapt_single(param_dict_hc, sd_dict_hc, group_name, seed=seed_num,num_sj=subj_num, num_trial=trial_num, model_name=model_name, plot=True)
    elif group_name == 'pt':
        # simulate pt subjects with given params
        sim_motoradapt_single(param_dict_pt, sd_dict_pt, group_name, seed=seed_num, num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    else:
        print('check group name (hc or pt)')