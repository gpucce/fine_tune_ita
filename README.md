# minerva_sft

This repo contains several utilities to finetune Decoder-only LLM on two different generative downstream tasks, **News Summarization** and **Machine Translation EN-IT|IT-EN**.

## Install

Installing this tool is very straight forward

``` sh
git clone git@github.com:Andrew-Wyn/minerva_sft.git

cd minerva_sft

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
````

### Run

To run a training lets do this on single GPU

````bash
python finetuning/finetune_summary.py -c configurations/mistral-base_continual.yaml
````

Run un 4 GPU with deepspeed setting 3

````bash
source /leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/.env/bin/activate

accelerate launch accelerate launch --multi_gpu --config_file=accelerate_configurations/deepspeed_zero3.yaml --num_processes 4 finetuning/finetune_summary.py -c configurations/mistral-base_continual.yaml
````

#### LORA

to run lora on single GPU

````sh
source /leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/.env/bin/activate

python finetuning/finetune_summary.py -c configurations/mistral-base_continual-lora.yaml
````

#### Utilities shell scripts

I added some shell scripts that can be invoked to run the slurm stuff, please substitute the paths and account stuff as you want

- finetuning/finetune_summary-lora.sh
- finetuning/finetune_summary-multigpu.sh
- finetuning/finetune_summary-multinode.sh

E.G.

````sh
sbatch finetuning/finetune_summary-lora.sh configurations/mistral-base_continual-lora.yaml
````