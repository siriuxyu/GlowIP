#!/bin/bash
#BSUB -J {JOBNAME}
#BSUB -q gpuq
#BSUB -n 16
#BSUB -R "span[ptile=4] select[ngpus>0] rusage[ngpus_shared=8]" 
#BSUB -o /gpfsdata/home/Zhaobo_hengjia21/GlowIP/results/output_{JOBNAME}.txt
#BSUB -e /gpfsdata/home/Zhaobo_hengjia21/GlowIP/results/errput_{JOBNAME}.txt

# Set environment variables
export PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/bin:$PATH
export PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/envs/gpu122/bin:$PATH
export LD_LIBRARY_PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/envs/gpu122/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH

# Change to project directory
cd /gpfsdata/home/Zhaobo_hengjia21/GlowIP
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu122

# Get the hostname of the master node
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Get node rank from LSB_JOBINDEX if available, otherwise default to 0
if [ -z "${LSB_JOBINDEX}" ]; then
    export NODE_RANK=0
else
    # LSB_JOBINDEX starts from 1, so subtract 1 for zero-based rank
    export NODE_RANK=$((LSB_JOBINDEX-1))
fi

# Set NCCL environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

echo "Starting training on node ${NODE_RANK} with master node ${MASTER_ADDR}"

torchrun \
  --nproc_per_node=2 \
  --nnodes=4 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_glow_ddp.py \
  -batchsize {BATCHSIZE} \
  -dataset {DATASET} \
  -size {SIZE} \
  -job_id {JOBID} \
  -epochs 800 \
  -n_data 800 \
  -lr 5e-5 \
  -coupling_bias {COUPLING_BIAS} \
  >> {LOGFILE}
