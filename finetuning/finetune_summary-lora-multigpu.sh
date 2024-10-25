# export OMP_PROC_BIND=true
export WANDB_MODE=offline

accelerate launch --config_file=./accelerate_configurations/fsdp_lora.yaml \
    --num_processes 4 \
    ./minerva_sft/finetuning/finetune_summary.py \
    -c $1