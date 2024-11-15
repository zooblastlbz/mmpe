#!/bin/bash
#SBATCH --job-name=without_mmpe
#SBATCH --output=/mnt/petrelfs/libozhou/mmpe/output/low_mmpe/finetune/%j.out
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=s2_bigdata

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0


NUM_GPUS=4
NNODES=1
RANK=0
LLM_VERSION="/mnt/hwfile/opendatalab/lbz/vicuna-7b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/mnt/hwfile/opendatalab/lbz/CLIP-ViT-L-14-laion2B-s32B-b82K"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

#4卡时batchsize=2,gradient_accumulation_steps=4,八卡时batchsize=4,gradient_accumulation_steps=1
############### Pretrain ################

PROMPT_VERSION="v1"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
PORT=$((RANDOM % 101 + 20000))
echo $ADDR
echo $PORT

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /mnt/hwfile/opendatalab/lbz/llava-sft/llava_v1_5_mix665k.json \
    --image_folder /mnt/hwfile/opendatalab/lbz/llava-sft \
    --pretrain_mm_mlp_adapter /mnt/petrelfs/libozhou/mmpe/output/low_mmpe/pretrain/mm_projector.bin \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --use_mmpe True \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(224, 448), (448, 224), (448, 448), (672, 224), (224, 672)]" \
    --mm_patch_merge_type spatial \
    --only_448 False \
    --bf16 True \
    --output_dir /mnt/petrelfs/libozhou/mmpe/output/low_mmpe/finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --run_name finetune_mmpe \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
