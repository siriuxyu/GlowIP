#!/bin/bash
#BSUB -J {JOBNAME}
#BSUB -q gpuq
#BSUB -n 4
#BSUB -gpu "num={NGPUS}"
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

export MASTER_ADDR=$(host $(echo $LSB_HOSTS | awk '{print $1}') | awk '{print $1}')
export MASTER_PORT=29500

HOSTNAMES=($(echo $LSB_HOSTS | tr ' ' '\n' | uniq))
NODE_RANK=0
for i in "${!HOSTNAMES[@]}"; do
  [[ "${HOSTNAMES[$i]}" == "$(hostname)" ]] && NODE_RANK=$i
done

torchrun \
  --nproc_per_node=2 \
  --nnodes={NNODE} \
  --node_rank=$NODE_RANK \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_glow.py \
  --batchsize {BATCHSIZE} \
  --dataset {DATASET} \
  --size {SIZE} \
  >> {LOGFILE} 2>&1

