# Adapted LLAMA using LORA

This repo contains several utilities to finetune Decoder-only LLM on two different generative downstream tasks, **News Summarization** and **Machine Translation EN-IT|IT-EN**.

## Install

Installing this tool is very straight forward

``` sh

pip install -r requirements.txt
```

## Usage

*For now **only News Summarization** is handled.*

### Configurations

When you want to start a new run, at first, you have to define a **configuration file**.

Some configurations are already present under the folder `configurations`.

E.g.

````yaml 
output_dir: "" # directory where the resulting model will be save, the last relative path will be used as indicato for wandb
model_name: "" # HF path or local dir of the model
use_lora: false # if needed
# Template
response_template: "###Summary:" # special token added to the model
prompt_template: "###Text:" # special token added to the model
# Training Parameters
training_bs: 16 # training batch size
evaluate_bs: 16 # evaluation batch size
num_train_epochs: 7
## Optimizer Parameters
weight_decay: 5e-3
learning_rate: 1e-5
lr_scheduler_type: "linear"
wermup_ratio: 0.3
max_source_len: 1000
max_target_len: 300
````

### Run

To run a training lets do this on multiple GPU

````bash
CUDA_VISIBLE_DEVICES="5,6" accelerate launch --config_file=./accelerate_configurations/fsdp_lora.yaml     --num_processes 2 --main_process_port=29501     ./finetuning/finetune_summary.py     -c /raid/homes/kajal.negi/workspace/fine_tune_ita/configurations/llama-base_continual-lora.yaml
````

To run on multiple GPU for evaluating original model
````bash

CUDA_VISIBLE_DEVICES="0,1,5,6" accelerate launch --config_file=./accelerate_configurations/fsdp_lora.yaml --main_process_port=29502 ./finetuning/evaluate_model.py -c /raid/homes/kajal.negi/workspace/fine_tune_ita/configurations/llama-base_continual-lora.yaml
````

To run on multiple GPU for evaluating adapted model

````bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="1,5,6" accelerate launch --config_file=./accelerate_configurations/fsdp_lora.yaml --main_process_port=29501 ./finetuning/evaluate_adapter_model.py -c /raid/homes/kajal.negi/workspace/fine_tune_ita/configurations/llama-adapter_continual-lora.yaml
````

