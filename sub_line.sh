#!/bin/bash
# job submission loop 

for sim_num in {1..50}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 250 -R 64 python power_bandit4arm_lapse.py pt $sim_num 80 200
done

for sim_num in {1..50}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 350 -R 64 python power_bandit4arm_lapse.py hc $sim_num 120 200
done