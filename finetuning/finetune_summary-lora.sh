#!/bin/bash
#SBATCH --job-name=finetuning_summary_mistral_base-lora            # Job name
#SBATCH -o logs/finetuning_summary_mistral_base-lora-job.out       # Name of stdout output file
#SBATCH -e logs/finetuning_summary_mistral_base-lora-job.err       # Name of stderr error file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=32              # number of threads per task
#SBATCH --time 00:15:00                 # format: HH:MM:SS
#SBATCH --gres=gpu:1                    # number of gpus per node


#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=/leonardo_scratch/large/userexternal/lmoroni0/hf_cache
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/large/userexternal/lmoroni0/instruct_tuning_voc_adapt/wandb_logs

source /leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/.env/bin/activate

accelerate launch --config_file=/leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/accelerate_configurations/deepspeed_zero2.yaml \
    --num_processes 1 \
    /leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/finetuning/finetune_summary.py \
    -c $1
    #-c /leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/configurations/mistral-base_continual-lora.yaml