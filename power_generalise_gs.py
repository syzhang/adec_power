"""
simulated power calculation for generalisation task (with gen in model)
"""
import sys
import pickle
import numpy as np
import pandas as pd
# from hbayesdm import rhat, print_fit, plot_hdi, hdi
# from hbayesdm.models import bandit4arm_lapse

def model_generalise_gs(param_dict, subjID, num_trial=200):
    """simulate shapes, go/nogo actions, and shock outcomes"""
    # load predefined image sequences (38 trials x 5 blocks)
    trial_type = 
    
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