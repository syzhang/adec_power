"""concat AMT fitting results"""

import sys, os
import pickle
import numpy as np
import pandas as pd

output_dir = './tmp_output/'

subj_out = []
for f in os.listdir(output_dir):
    f_path = os.path.join(output_dir, f)
    df_tmp = pd.read_csv(f_path)
    
    # check individual fitted params only
    cols = df_tmp.columns
    cols_pick = ['sigma_a','sigma_n','eta','kappa','beta','bias']
    for idx, c in enumerate(cols_pick):
        out = []
        for i in range(1,12):
            c_name = f'{c}[{i}]'
            if c_name in df_tmp:
                c_mean = df_tmp[c_name].mean()
                c_std = df_tmp[c_name].std()
                out.append(pd.DataFrame({c+'_mean':[c_mean], c+'_std':[c_std]}))
        c_tmp = pd.concat(out)
        if idx ==0:
            c_out = c_tmp.reset_index(drop=True)
        else:
            c_out = c_out.join(c_tmp.reset_index(drop=True))
    subj_out.append(c_out)
df_subj = pd.concat(subj_out)

df_subj.to_csv('./AMT_subj_params.csv',index=None)