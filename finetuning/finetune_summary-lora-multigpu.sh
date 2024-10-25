# export OMP_PROC_BIND=true
export WANDB_MODE=offline

accelerate launch --config_file=/leonardo_scratch/large/userexternal/gpuccett/Repos/minerva_sft/accelerate_configurations/fsdp_lora.yaml \
    --num_processes 4 \
    /leonardo_scratch/large/userexternal/gpuccett/Repos/minerva_sft/finetuning/finetune_summary.py \
    -c $1
    #-c /leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/configurations/mistral-base_continual-lora.yaml