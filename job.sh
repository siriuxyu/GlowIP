#!/bin/bash
#BSUB -J test
#BSUB -q gpuq
#BSUB -n 4
#BSUB -R "select[ngpus>0]rusage [ngpus_shared=2]"
#BSUB -o /gpfsdata/home/Zhaobo_hengjia21/GlowIP/output.txt
#BSUB -e /gpfsdata/home/Zhaobo_hengjia21/GlowIP/errput.txt

nvidia-smi
export PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/bin:$PATH
export PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/envs/gpu122/bin:$PATH
export LD_LIBRARY_PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/envs/gpu122/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH


cd /gpfsdata/home/Zhaobo_hengjia21/GlowIP

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu122

python solve_cs.py -experiment exp2 -dataset ood -prior glow -model celeba \
-m 5000 -gamma 0 -init_std 0 -device cuda >> ip_log.txt
