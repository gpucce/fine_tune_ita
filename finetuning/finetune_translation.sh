#!/bin/bash
#SBATCH --job-name=finetuning_translation_350M            # Job name
#SBATCH -o logs/finetuning_translation_350M-job.out       # Name of stdout output file
#SBATCH -e logs/finetuning_translation_350M-job.err       # Name of stderr error file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=32              # number of threads per task
#SBATCH --time 08:00:00                  # format: HH:MM:SS
#SBATCH --gres=gpu:4                    # number of gpus per node


#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=/leonardo_scratch/large/userexternal/lmoroni0/hf_cache
export WANDB_MODE=offline

source /leonardo/home/userexternal/lmoroni0/__Work/llm_summarization_finetunig/env/bin/activate

accelerate launch /leonardo/home/userexternal/lmoroni0/__Work/llm_summarization_finetunig/finetuning/finetune_translation.py --num_processes 4


