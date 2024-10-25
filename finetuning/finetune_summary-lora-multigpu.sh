#!/bin/bash
#SBATCH --job-name=finetuning_summary_mistral_base-lora            # Job name
#SBATCH --output /leonardo_scratch/large/userexternal/gpuccett/Repos/minerva_sft/slurm_logs/sft_%j.out       # Name of stdout output file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=32              # number of threads per task
#SBATCH --time 12:00:00                 # format: HH:MM:SS
#SBATCH --gres=gpu:4                    # number of gpus per node
#SBATCH -A EUHPC_E03_068
#SBATCH -p boost_usr_prod

module load profile/deeplrn
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

eval "$(conda shell.bash hook)" # init conda
conda activate /leonardo_scratch/large/userexternal/gpuccett/Repos/minerva_sft/conda_venv

# export OMP_PROC_BIND=true
export WANDB_MODE=offline

accelerate launch --config_file=/leonardo_scratch/large/userexternal/gpuccett/Repos/minerva_sft/accelerate_configurations/fsdp_lora.yaml \
    --num_processes 4 \
    /leonardo_scratch/large/userexternal/gpuccett/Repos/minerva_sft/finetuning/finetune_summary.py \
    -c $1
    #-c /leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/configurations/mistral-base_continual-lora.yaml