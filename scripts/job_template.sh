#!/bin/bash
#BSUB -J {JOBNAME}
#BSUB -q gpuq
#BSUB -n 4
#BSUB -R "select[ngpus>0] rusage[ngpus_shared=2]"
#BSUB -o /gpfsdata/home/Zhaobo_hengjia21/GlowIP/results/output_{JOBNAME}.txt
#BSUB -e /gpfsdata/home/Zhaobo_hengjia21/GlowIP/results/errput_{JOBNAME}.txt

export PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/bin:$PATH
export PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/envs/gpu122/bin:$PATH
export LD_LIBRARY_PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/envs/gpu122/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH

cd /gpfsdata/home/Zhaobo_hengjia21/GlowIP
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu122

python train_glow.py \
  -batchsize {BATCHSIZE} \
  -dataset {DATASET} \
  -size {SIZE} \
  -job_id {JOBID} \
  -epochs 10 \
  -n_data 80 \
  >> {LOGFILE}
