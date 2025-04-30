#!/bin/bash
#SBATCH --job-name=finetuning_summary_mistral_base            # Job name
#SBATCH -o logs/finetuning_summary_mistral_base-job.out       # Name of stdout output file
#SBATCH -e logs/finetuning_summary_mistral_base-job.err       # Name of stderr error file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=32              # number of threads per task
#SBATCH --time 00:15:00                  # format: HH:MM:SS
#SBATCH --gres=gpu:4                    # number of gpus per node


#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

#module load profile/deeplrn
#module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=/home/kajalnegi/raid/hf_cache
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="5"
source /home/kajalnegi/anaconda3/bin/python3

accelerate launch --config_file=/home/kajalnegi/raid/fine_tune_ita/accelerate_configurations/deepspeed_zero2.yaml --num_processes 4 --multi_gpu \
    /home/kajalnegi/raid/fine_tune_ita/finetuning/finetune_summary.py \
    -c $1
    #-c /home/kajalnegi/raid/fine_tune_ita/configurations/llama-base_continual-lora.yaml


