"""
fit generalisation instrumetnal avoidance task with AMT data
"""
import sys, os
import pickle
import numpy as np
import pandas as pd
import pystan
from hbayesdm.models import generalise_gs
from hbayesdm import rhat, print_fit

def split_txt(in_file='./AMT_behavioural.txt', num_files=160):
    """split data to run in parallel"""
    df = pd.read_csv(in_file, sep='\t')
    # print(df.head())
    num_sj = len(np.unique(df['subjID']))
    batch_sj = np.ceil(num_sj/num_files)
    # split save
    output_dir = './split_files'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # split loop
    subj_start = 0 
    for f in range(num_files):
        df_tmp = df[(df['subjID']>=subj_start) & (df['subjID']<subj_start+batch_sj)]
        print(np.unique(df_tmp['subjID']))
        subj_start += batch_sj
        print(df_tmp.shape)
        df_tmp.to_csv(f'./split_files/AMT_{f:02d}.txt', sep='\t', index=None)


if __name__ == "__main__":
    # split data
    # split_txt(num_files=44)

    # parsing cl arguments
    # fc = int(sys.argv[1]) # file count

    # # fit
    # # Run the model and store results in "output"
    # output = generalise_gs(f'./split_files/AMT_{fc:02d}.txt', niter=3000, nwarmup=1500, nchain=4, ncore=16)

    # # using pystan.misc.to_dataframe -function works also with the older fit objects
    # df = pystan.misc.to_dataframe(output.fit)
    # df.to_csv(f'./tmp_output/AMT_{fc:02d}.csv', index=None)

    # # sub
    # # fsl_sub -T 30 -R 64 python fit_AMT.py 0
    # for sim_num in {0..43}
    # do
    # echo "submitted job simulation with seed $sim_num "
    # fsl_sub -T 30 -R 64 python fit_AMT.py $sim_num
    # done

    #######################################################
    # # fit only a single file
    output = generalise_gs('./AMT_behavioural.txt', niter=3000, nwarmup=1500, nchain=4, ncore=16)

    # using pystan.misc.to_dataframe -function works also with the older fit objects
    df = pystan.misc.to_dataframe(output.fit)
    df.to_csv('./tmp_output/AMT_behavioural.csv', index=None)
    # # fsl_sub -T 30*50 -R 64 python fit_AMT.py