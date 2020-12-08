"""
simulated power calculation for bandit 4-arm task (with lapse in model)
"""
import os, sys
import pickle
import numpy as np
import pandas as pd
from hbayesdm.models import bandit4arm_lapse

def sim_bandit4arm_lapse(param_dict, sd_dict, group_name, seed, 
                         num_sj=50, num_trial=200,
                         model_name='bandit4arm_lapse'):
    """simulate 4 arm bandit data for multiple subjects"""
    multi_subject = []
    
    # generate new params
    np.random.seed(seed)
    sample_params = dict()
    for key in param_dict:
        sample_params[key] = np.random.normal(param_dict[key], sd_dict[key], size=1)[0]
    
    for sj in range(num_sj):
        df_sj = model_bandit4arm_lapse(param_dict, sj, num_trial)
        multi_subject.append(df_sj)
        
    df_out = pd.concat(multi_subject)
    # saving output
    output_dir = './tmp_output/bandit_sim/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f_name = model_name+'_'+group_name+'_'+str(seed)
    df_out.to_csv(output_dir+f_name+'.txt', sep='\t', index=False)
    print(df_out)
    # return df_out

def model_bandit4arm_lapse(param_dict, subjID, num_trial=200):
    """simulate 4-arm bandit choices and outcomes"""
    # load reward/pain probabilities
    pun_prob = pd.read_csv('./pain_prob.csv').values
    rew_prob = pd.read_csv('./reward_prob.csv').values
    
    # initialise values
    Qr = np.zeros(4)
    Qp = np.zeros(4)
    
    # initial probabilities of choosing each deck
    pD = 0.25 * np.ones(4)
    
    # initialise output
    data_out = []
    # simulate trials
    for t in range(num_trial):
        # select a deck
        tmpDeck = int(np.random.choice(np.arange(4), size=1, p=pD, replace=True))

        # compute tmpRew and tmpPun
        tmpRew = np.random.choice([0,1], size=1, replace=True, 
                                  p=[1-rew_prob[t,tmpDeck], rew_prob[t, tmpDeck]])
        tmpPun = np.random.choice([0,-1], size=1, replace=True, 
                                  p=[1-pun_prob[t,tmpDeck], pun_prob[t, tmpDeck]])

        # compute PE and update values
        PEr = param_dict['R']*tmpRew - Qr[tmpDeck]
        PEp = param_dict['P']*tmpPun - Qp[tmpDeck]
        PEr_fic = -Qr
        PEp_fic = -Qp
        
        Qr_chosen, Qp_chosen = [], []
        Qr_chosen = Qr[tmpDeck]
        Qp_chosen = Qp[tmpDeck]
        
        # update Qr and Qp
        Qr += param_dict['Arew'] * PEr_fic
        Qp += param_dict['Apun'] * PEp_fic
        # replace Q values of chosen deck with correct values
        Qr[tmpDeck] = Qr_chosen + param_dict['Arew'] * PEr
        Qp[tmpDeck] = Qp_chosen + param_dict['Apun'] * PEp
        
        # sum Q
        Qsum = Qr + Qp
        
        # normalise to avoid overflow
        if sum(Qsum) != 0.:
            Qsum_norm = Qsum / Qsum.sum()
        else: 
            Qsum_norm = Qsum
            
        # update pD for next trial
        pD_pre = np.exp(Qsum_norm) / sum(np.exp(Qsum_norm))

        # xi/lapse
        pD = pD_pre * (1.-param_dict['xi']) + param_dict['xi']/4.
        
        # output
        data_out.append([subjID, t, tmpDeck+1, int(tmpRew), int(tmpPun)])
        
    df_out = pd.DataFrame(data_out)
    df_out.columns = ['subjID', 'trial', 'choice', 'gain', 'loss']
#     print(df_out)
    return df_out

if __name__ == "__main__":
    # healthy control parameters
    param_dict_hc = {
        'Arew': 9.61,  # reward sensitivity
        'Apun': 6.67,  # punishment sensitivity
        'R':    0.25,  # reward learning rate
        'P':    0.31,  # punishment learning rate
        'xi':   0.13   # lapse
    }

    # assumed patient parameters (based on Aylward 2019)
    param_dict_pt = {
        'Arew': 7.47,  # reward sensitivity
        'Apun': 7.41,  # punishment sensitivity
        'R':    0.31,  # reward learning rate
        'P':    0.51,  # punishment learning rate
        'xi':   0.21   # lapse
    }

    # healthy control parameter sd
    # sd_dict_hc = {
    #     'Arew': 4.87,  # reward sensitivity
    #     'Apun': 4.83,  # punishment sensitivity
    #     'R':    0.22,  # reward learning rate
    #     'P':    0.15,  # punishment learning rate
    #     'xi':   0.11   # lapse
    # }
    # assumed lower variance (same with patients) 
    sd_dict_hc= {
        'Arew': 2.91,  # reward sensitivity
        'Apun': 2.21,  # punishment sensitivity
        'R':    0.1,  # reward learning rate
        'P':    0.1,  # punishment learning rate
        'xi':   0.05   # lapse
    }

    # assumed patient parameters (based on Aylward 2019)
    # sd_dict_pt = {
    #     'Arew': 2.91,  # reward sensitivity
    #     'Apun': 7.21,  # punishment sensitivity
    #     'R':    0.30,  # reward learning rate
    #     'P':    0.18,  # punishment learning rate
    #     'xi':   0.10   # lapse
    # }
    # assumed lower variance in patients 
    sd_dict_pt = {
        'Arew': 2.91,  # reward sensitivity
        'Apun': 2.21,  # punishment sensitivity
        'R':    0.1,  # reward learning rate
        'P':    0.1,  # punishment learning rate
        'xi':   0.05   # lapse
    }

    # parsing cl arguments
    group_name = sys.argv[1] # pt=patient, hc=control
    seed_num = int(sys.argv[2]) # seed number
    subj_num = int(sys.argv[3]) # subject number to simulate
    trial_num = int(sys.argv[4]) # trial number to simulate

    model_name = 'bandit4arm_lapse'
    if group_name == 'hc':
        # simulate hc subjects with given params
        sim_bandit4arm_lapse(param_dict_hc, sd_dict_hc, group_name, seed=seed_num,num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    elif group_name == 'pt':
        # simulate pt subjects with given params
        sim_bandit4arm_lapse(param_dict_pt, sd_dict_pt, group_name, seed=seed_num, num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    else:
        print('check group name (hc or pt)')

    # fit
    # Run the model and store results in "output"
    output = bandit4arm_lapse('./tmp_output/bandit_sim/'+model_name+'_'+group_name+'_'+str(seed_num)+'.txt', niter=2000, nwarmup=1000, nchain=4, ncore=1)

    # debug
    print(output.fit)

    # saving
    sfile = './tmp_output/bandit_sim/'+group_name+'_sim_'+str(seed_num)+'.pkl'
    with open(sfile, 'wb') as op:
        tmp = { k: v for k, v in output.par_vals.items() if k in ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi'] } # dict comprehension
        pickle.dump(tmp, op)


