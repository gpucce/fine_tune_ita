# export OMP_PROC_BIND=true
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="3,4,6"

PYTHONPATH=/home/kajalnegi/anaconda3/bin/python3
export PYTHONPATH
accelerate launch --config_file="/home/kajalnegi/raid/fine_tune_ita/accelerate_configurations/fsdp_lora.yaml" \
    --num_processes 4 \
    --main_process_port=29501 \
    /home/kajalnegi/raid/fine_tune_ita/finetuning/finetune_summary.py \
    -c $1 
    