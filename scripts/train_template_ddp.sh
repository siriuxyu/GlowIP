#!/bin/bash
#BSUB -J {JOBNAME}
#BSUB -q gpuq
#BSUB -n 8
#BSUB -R "span[ptile=4] select[ngpus>0] rusage[ngpus_shared=4]" 
#BSUB -o /gpfsdata/home/Zhaobo_hengjia21/GlowIP/results/output_{JOBNAME}.txt
#BSUB -e /gpfsdata/home/Zhaobo_hengjia21/GlowIP/results/errput_{JOBNAME}.txt
#BSUB -W 24:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu122
cd /gpfsdata/home/Zhaobo_hengjia21/GlowIP


# -------- 解析节点 ----------
if [[ -n "$LSB_DJOB_HOSTFILE" && -s "$LSB_DJOB_HOSTFILE" ]]; then
    NODES=( $(sort -u "$LSB_DJOB_HOSTFILE") )
else
    # LSB_HOSTS 是空格分隔字符串：gpu06 gpu06 gpu04 …
    NODES=( $(echo "$LSB_HOSTS" | tr ' ' '\n' | sort -u) )
fi

GPUS_PER_NODE=2
NUM_NODES=${#NODES[@]}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0   # 如无 RDMA，可注释
export OMP_NUM_THREADS=4

echo "[DEBUG] nodes=${NODES[*]}"
if [[ ${#NODES[@]} -eq 0 ]]; then
    echo "[ERROR] 解析节点失败，脚本终止"; exit 1
fi

echo "[DDP] MASTER=$MASTER_ADDR:$MASTER_PORT  WORLD_SIZE=$WORLD_SIZE"

# ---------- 在每个节点启动 torchrun ----------
for idx in "${!NODES[@]}"; do
  host=${NODES[$idx]}
  NODE_RANK=$idx

  blaunch -z ${host} \
    torchrun \
      --nnodes ${NUM_NODES} \
      --nproc_per_node ${GPUS_PER_NODE} \
      --node_rank ${NODE_RANK} \
      --master_addr ${MASTER_ADDR} \
      --master_port ${MASTER_PORT} \
      train_glow_ddp.py \
        -batchsize {BATCHSIZE} \
        -dataset {DATASET} \
        -size {SIZE} \
        -job_id {JOBID} \
        -coupling_bias {COUPLING_BIAS} \
      >> results/{JOBNAME}_node${NODE_RANK}.log 2>&1 &
done
wait
