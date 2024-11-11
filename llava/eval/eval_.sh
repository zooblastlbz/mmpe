#!/bin/bash

python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=/mnt/petrelfs/libozhou/mmpe/output/direct_finetune_base_ \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --verbosity DEBUG \
    --log_samples_suffix llava_v1.5_mme \
    --output_path /mnt/petrelfs/libozhou/mmpe/output/direct_finetune_base/eval
#python /mnt/petrelfs/libozhou/VLMEvalKit/run.py --data MMBench_DEV_EN  --model llava_without_mmpe --work-dir /mnt/petrelfs/libozhou/mmpe/output/direct_finetune_direct_match_llava/eval