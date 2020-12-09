#!/bin/bash
# job submission loop 

# fixing gcc problem (install clang)
#conda install clang_osx-64 clangxx_osx-64 -c anaconda

#run sims for bandit
for sim_num in {1..50}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 350 -R 64 python power_bandit4arm_lapse.py pt $sim_num 200 200
done

for sim_num in {1..50}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 350 -R 64 python power_bandit4arm_lapse.py hc $sim_num 250 200
done

# hack conda environment
cp -r ../hBayesDM/commons/* /home/fs0/syzhang/.conda/envs/pystan/lib/python3.9/site-packages/hbayesdm/common/

# run sims for generalise
for sim_num in {0..50}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 300 -R 64 python power_generalise_gs.py pt $sim_num 70 190
done

for sim_num in {0..50}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 300 -R 64 python power_generalise_gs.py hc $sim_num 90 190
done