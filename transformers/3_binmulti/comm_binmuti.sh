#!/bin/bash
#SBATCH -J finetune_binmulti
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -o /home/mexposit/cg/gea/transformers/3_binmulti/logs/finetune_3binmulti.log
#SBATCH -e /home/mexposit/cg/gea/transformers/3_binmulti/logs/finetune_3binmulti.err

# activate conda and environment for training
source ~/.bashrc
conda activate dnabert

# user defined params
# can be path to directory that contains config.json file inside or directly to config.json file
PATH_TO_THE_PRETRAINED_MODEL='/home/mexposit/cg/gea/dnabert/model/pretrained/6-new-12w-0'
KMER_VAL=6
DATA_PATH_ABS='/home/mexposit/cg/gea/transformers/3_binmulti/in_data'
OUT_DIR='/home/mexposit/cg/gea/transformers/3_binmulti/model/ft_3binmulti'

export KMER=$KMER_VAL
export MODEL_PATH=$PATH_TO_THE_PRETRAINED_MODEL
export DATA_PATH=$DATA_PATH_ABS
export OUTPUT_PATH=$OUT_DIR

python /home/mexposit/cg/gea/transformers/trans_utils/run_finetune_mod.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 100 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 10 \
    --save_steps 500 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
