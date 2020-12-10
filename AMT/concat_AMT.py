"""concat AMT fitting results"""

import sys, os
import pickle
import numpy as np
import pandas as pd

output_dir = './tmp_output/'

for f in os.listdir(output_dir):
    if '00' in f:
        f_path = os.path.join(output_dir, f)
        df_tmp = pd.read_csv(f_path)
        
        print(df_tmp.mean())