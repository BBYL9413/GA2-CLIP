#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi
export TORCH_DISTRIBUTED_DEBUG=DETAIL
now=$(date +"%Y%m%d_%H%M%S")


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 14237 --nproc_per_node=8 \
         train_nce.py  --config ${config} --log_time $now