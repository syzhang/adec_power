"""
simulated power calculation for generalisation instrumetnal avoidance task (with gen in model)
"""
import sys, os
import pickle
import numpy as np
import pandas as pd
from hbayesdm.models import generalise_gs
# from hbayesdm import rhat, print_fit, plot_hdi, hdi

def sigmoid(p):
    return 1./(1.+np.exp(-p))

def softmax_perception(s_cue, Q, beta, bias):
    """probability of avoidance given stim and params"""
    avoid_cost = 0.2 # fixed
    if s_cue == 1:
        pred_V = 0.75*Q[s_cue-1] + 0.25*Q[2-1]
    elif s_cue == 3:
        pred_V = 0.75*Q[s_cue-1] + 0.25*Q[2-1]
    elif s_cue == 5:
        pred_V = 0.75*Q[s_cue-1] + 0.25*Q[6-1]
    elif s_cue == 7:
        pred_V = 0.75*Q[s_cue-1] + 0.25*Q[6-1]
    else:
        pred_V = Q[s_cue-1]

    # avoidance probability
    gx = 1./(1. + np.exp(-beta*(0. - pred_V - avoid_cost - bias)))
    return gx    

def draw_cue(num_trial):
    """drawing cues based on given probability"""
    trial_type = []
    p_cues = [0.052, 0.264, 0.052, 0.264,
            0.052, 0.264, 0.052]
    for i in range(num_trial):
        smp = np.random.choice(7, p=p_cues)
        trial_type.append(smp+1) # match matlab number
    return trial_type

def model_generalise_gs(param_dict, subjID, num_trial=190):
    """simulate shapes, avoid actions, and shock outcomes"""
    # load predefined image sequences (38 trials x 5 blocks)
    trial_type = np.squeeze(pd.read_csv('./generalise_stim.csv').values) # 1:7
    # trial_type = draw_cue(num_trial)
    num_state = len(np.unique(trial_type))

    # initialise values (values from AN's VBA code)
    # Q = sigmoid(-0.2) * np.ones(num_state) # 7 possible cues
    Q = np.zeros(num_state) # 7 possible cues
    alpha = sigmoid(0.95) * np.ones(1) # initial learning rate
    
    # initialise output
    data_out = []

    # simulate trials
    for t in range(num_trial):
        # a cue is shown (not sure how it's generated)
        s_cue = int(trial_type[t])

        # avoid or not
        p_avoid = softmax_perception(s_cue, Q, param_dict['beta'], param_dict['bias'])
        a = int(np.random.binomial(size=1, n=1, p=p_avoid))
        # a = int(np.random.choice([0,1], size=1, 
        #     p=[1-p_avoid, p_avoid], replace=True))
        # print(p_avoid, a)

        # deliver shock or not
        if (s_cue == 2 or s_cue == 4) and a == 0: # if CS+1 or 2 shock trials and no avoidance made
            r = -1
        else:
            r = 0

        # define sensory params
        # mean_theta = 0.25
        # rhos = [0.25-mean_theta, 0.25, 0.25+mean_theta, 0.75-2*mean_theta, 0.75-mean_theta, 0.75, 0.75+mean_theta]
        rhos = np.array([0.0, 0.25, 0.5, 0.25, 0.5, 0.75, 1.0])

        # compute PE and update values
        if a == 0: # did not avoid
            # PE update
            PE = r - Q[s_cue-1]

            # define sigma
            if r == 0: # no shock
                sigma_t = param_dict['sigma_n']
            else: # shock
                sigma_t = param_dict['sigma_a']

            # update Q
            # for s in range(num_state):
            #     rho = rhos[s]
            #     G = 1./np.exp((rho-cue_rho)**2. / (2.*sigma_t**2.))
            #     Q[s] += param_dict['kappa'] * alpha * PE * G
            # current cue rho value
            cue_rho = rhos[s_cue-1] * np.ones(num_state)
            diff2 = (rhos-cue_rho)**2. / (2.*(sigma_t**2.))
            G = 1./np.exp(diff2)
            Q += param_dict['kappa'] * alpha * PE * G

        else: # avoided
            PE = 0.
            Q = Q

        # update alpha
        alpha = param_dict['eta']*np.abs(PE) + (1-param_dict['eta'])*alpha

        # output
        data_out.append([subjID, t, s_cue, a, r])
        
    df_out = pd.DataFrame(data_out)
    df_out.columns = ['subjID', 'trial', 'cue', 'choice', 'outcome']
    # print(df_out)
    return df_out

def sim_generalise_gs(param_dict, sd_dict, group_name, seed, 
                         num_sj=50, num_trial=190, model_name='generalise_gs'):
    """simulate generalise instrumental avoidance task for multiple subjects"""
    multi_subject = []
    
    # generate new params
    np.random.seed(seed)
    sample_params = dict()
    for key in param_dict:
        sample_params[key] = np.random.normal(param_dict[key], sd_dict[key], size=1)[0]
    
    for sj in range(num_sj):
        df_sj = model_generalise_gs(sample_params, sj, num_trial)
        multi_subject.append(df_sj)
        
    df_out = pd.concat(multi_subject)
    # saving output
    output_dir = './tmp_output/generalise_sim/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f_name = model_name+'_'+group_name+'_'+str(seed)
    df_out.to_csv(output_dir+f_name+'.txt', sep='\t', index=False)
    print(df_out)

if __name__ == "__main__":
    # parameters from paper
    param_dict_hc = {
        'sigma_a': 0.75,  # generalisation param for shock
        'sigma_n': 0.028,  # generalisation param for no shock
        'eta':    0.5,     # p_h dynamic learning rate
        'kappa':  0.4,    # p_h dynamic learning rate
        'beta': 1.,       # softmax beta
        'bias': 0.5      # softmax bias
    }
    # parameters from paper
    param_dict_pt = {
        'sigma_a': 0.15,  # generalisation param for shock
        'sigma_n': 0.088,  # generalisation param for no shock
        'eta':    0.2,     # p_h dynamic learning rate
        'kappa':  0.5,    # p_h dynamic learning rate
        'beta': 1.5,       # softmax beta
        'bias': 0.7      # softmax bias
    }
    # sd from paper
    sd_dict_hc = {
        'sigma_a': 0.29,  # generalisation param for shock
        'sigma_n': 0.03,  # generalisation param for no shock
        'eta':    0.01,     # p_h dynamic learning rate
        'kappa':  0.05,    # p_h dynamic learning rate
        'beta': 0.05,       # softmax beta
        'bias': 0.05      # softmax bias
    }
    # sd from paper
    sd_dict_pt = {
        'sigma_a': 0.29,  # generalisation param for shock
        'sigma_n': 0.03,  # generalisation param for no shock
        'eta':    0.01,     # p_h dynamic learning rate
        'kappa':  0.05,    # p_h dynamic learning rate
        'beta': 0.05,       # softmax beta
        'bias': 0.05      # softmax bias
    }

    # parsing cl arguments
    group_name = sys.argv[1] # pt=patient, hc=control
    seed_num = int(sys.argv[2]) # seed number
    subj_num = int(sys.argv[3]) # subject number to simulate
    trial_num = int(sys.argv[4]) # trial number to simulate

    model_name = 'generalise_gs'
    if group_name == 'hc':
        # simulate hc subjects with given params
        sim_generalise_gs(param_dict_hc, sd_dict_hc, group_name, seed=seed_num,num_sj=subj_num, model_name=model_name)
    elif group_name == 'pt':
        # simulate pt subjects with given params
        sim_generalise_gs(param_dict_pt, sd_dict_pt, group_name, seed=seed_num, num_sj=subj_num, model_name=model_name)
    else:
        print('check group name (hc or pt)')

    # fit
    # Run the model and store results in "output"
    output = generalise_gs('./tmp_output/generalise_sim/'+model_name+'_'+group_name+'_'+str(seed_num)+'.txt', niter=3000, nwarmup=1500, nchain=4, ncore=16)

    # debug
    print(output.fit)

    # saving
    sfile = './tmp_output/generalise_sim/'+group_name+'_sim_'+str(seed_num)+'.pkl'
    with open(sfile, 'wb') as op:
        tmp = { k: v for k, v in output.par_vals.items() if k in ['mu_sigma_a', 'mu_sigma_n', 'mu_eta', 'mu_kappa', 'mu_beta', 'mu_bias'] } # dict comprehension
        pickle.dump(tmp, op)