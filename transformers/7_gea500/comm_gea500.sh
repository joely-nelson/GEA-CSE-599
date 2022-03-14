#!/bin/bash
#SBATCH -J finetune_7gea500
#SBATCH -p gpu
#SBATCH --gres=gpu:quadro:1
#SBATCH --mem=48G
#SBATCH -c 2
#SBATCH -o /home/mexposit/cg/gea/transformers/7_gea500/logs/finetune_7gea500.log
#SBATCH -e /home/mexposit/cg/gea/transformers/7_gea500/logs/finetune_7gea500.err

# activate conda and environment for training
source ~/.bashrc
conda activate dnabert

# user defined params
# can be path to directory that contains config.json file inside or directly to config.json file

export KMER=6
export MODEL_PATH='/home/mexposit/cg/gea/dnabert/model/pretrained/6-new-12w-0'
export DATA_PATH='/home/mexposit/cg/gea/transformers/7_gea500/in_data'
export OUTPUT_PATH='/home/mexposit/cg/gea/transformers/7_gea500/model/ft_7_gea500'
export TASK='dnageaall'
export MAX_SEQ_LEN=512
export MODEL_TYPE='dna'

#MOD use the modified version /home/mexposit/cg/gea/transformers/trans_utils/run_finetune_mod.py that supports saving probs and top10accuracy
#MOD use --save_eval_probs to keep probability predictions
#MOD uses the task dnagea30 instead of dnaprom to support 30 different labels and also output top10 accuracy results (because I modified glue.py)
#MOD use --multiclass to keep predictions as (nsample,nclasses) and not only (nsample,) which is required for AUC and Top10 accuracy calculation

# this one takes twice: 300secs/iteration!! 

# This is the logic I used to decide logs and checkpoints
# 7500 training data sequences:
#     7500/32*5 = aprox 1170 steps
#     for longer sequences it takes about 160secs/iteration! his is very long compared to the 20 secs I had before
#     if I log every 10 steps, this is one every 30mins
#     evaluations must also be longer, so I reduce them quite a bit

python /home/mexposit/cg/gea/transformers/trans_utils/run_finetune_mod.py \
    --model_type $MODEL_TYPE \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name $TASK \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length $MAX_SEQ_LEN \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 20 \
    --save_steps 20 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --save_eval_probs \
    --multiclass \
    --weight_decay 0.01 \
    --n_process 8
