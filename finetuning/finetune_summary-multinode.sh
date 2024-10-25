#!/bin/bash
#SBATCH --job-name=finetuning_summary_minerva_3B            # Job name
#SBATCH -o logs/finetuning_summary_minerva_3B_sarti-job.out       # Name of stdout output file
#SBATCH -e logs/finetuning_summary_minerva_3B_sarti-job.err       # Name of stderr error file
#SBATCH --nodes=2                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=32              # number of threads per task
#SBATCH --time 08:00:00                  # format: HH:MM:SS
#SBATCH --gres=gpu:4                    # number of gpus per node


#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8


######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

# export OMP_PROC_BIND=true
# export HF_DATASETS_CACHE=/leonardo_scratch/large/userexternal/lmoroni0/hf_cache
export WANDB_MODE=offline

source /leonardo/home/userexternal/lmoroni0/__Work/llm_summarization_finetunig/env/bin/activate    

export LAUNCHER="accelerate launch \
    --num_processes 8 \
    --num_machines 2 \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank 0
    "
export SCRIPT="/leonardo/home/userexternal/lmoroni0/__Work/llm_summarization_finetunig/finetuning/finetune_summary.py"
export SCRIPT_ARGS=""
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $ARGS" 
srun $CMD