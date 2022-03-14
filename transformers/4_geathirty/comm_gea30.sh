#!/bin/bash
#SBATCH -J finetune_geathrity
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -o /home/mexposit/cg/gea/transformers/4_geathirty/logs/finetune_4geathirty.log
#SBATCH -e /home/mexposit/cg/gea/transformers/4_geathirty/logs/finetune_4geathirty.err

# activate conda and environment for training
source ~/.bashrc
conda activate dnabert

# user defined params
# can be path to directory that contains config.json file inside or directly to config.json file

export KMER=6
export MODEL_PATH='/home/mexposit/cg/gea/dnabert/model/pretrained/6-new-12w-0'
export DATA_PATH='/home/mexposit/cg/gea/transformers/4_geathirty/in_data'
export OUTPUT_PATH='/home/mexposit/cg/gea/transformers/4_geathirty/model/ft_4_geathirty'
export TASK='dnagea30'

#MOD use the modified version /home/mexposit/cg/gea/transformers/trans_utils/run_finetune_mod.py that supports saving probs and top10accuracy
#MOD use --save_eval_probs to keep probability predictions
#MOD uses the task dnagea30 instead of dnaprom to support 30 different labels and also output top10 accuracy results (because I modified glue.py)
#MOD use --multiclass to keep predictions as (nsample,nclasses) and not only (nsample,) which is required for AUC and Top10 accuracy calculation


# This is the logic I used to decide logs and checkpoints
# 7500 training data sequences:
#     7500/32*5 = aprox 1170 steps
#     training steps are 20 seconds, evaluation steps are 6seconds
#     if each step is 20 sec this is aprox 7hours
#     if I log every 25 steps, I get 45 logs, one every 8mins
#     Each log takes aprox 3 mins, so this adds up to 2 hours. It is quite a bit...but I don't think I will need to finish all process
#     I will make a checkpoint every 100 steps, I don't think I will need to finish the 5 epochs to get good results


python /home/mexposit/cg/gea/transformers/trans_utils/run_finetune_mod.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name $TASK \
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
    --logging_steps 25 \
    --save_steps 100 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --save_eval_probs \
    --multiclass \
    --weight_decay 0.01 \
    --n_process 8
