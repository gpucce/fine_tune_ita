from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import numpy as np
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model  
import torch
import numpy as np
import evaluate
from random import sample
import nltk

# VARS
FROM_LANG="en"
TO_LANG="it"
MODEL_SIZE="3B"
RUN_NAME=f"minerva_translation_{FROM_LANG}_{TO_LANG}_{MODEL_SIZE}"
CACHE_DATASETS="/leonardo_scratch/large/userexternal/lmoroni0/hf_cache"
OUTPUT_DIR=f"/leonardo_scratch/large/userexternal/lmoroni0/minerva_translation_it_en/{RUN_NAME}"
MODEL_NAME = f"sapienzanlp/Minerva-{MODEL_SIZE}-base-v1.0"
USE_LORA = False
EVALUATION_SAMPLES = 500
# ----

## EVALUATION FUNCTIONS

# Metric
metric = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    print(preds)
    print(labels)

    exit()

    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# ---

# DATASET
# News dataset is defined as union of Fanpage and IlPost
dataset_translation = load_dataset("Helsinki-NLP/opus-100", cache_dir=CACHE_DATASETS)

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DATASETS)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = 'right'

# FORMATTING
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['source'])):
        text = f"### Text: {example['translation'][i][TO_LANG]}\n ### Translation: {example['translation'][i][TO_LANG]}{tokenizer.eos_token}"
        output_texts.append(text)
    return output_texts

response_template = " ### Translation:"
prompt_template = "### Text:"
initial_token_count = len(tokenizer)
added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [prompt_template, response_template]})

# MODEL
if USE_LORA: ## LORA:

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                            cache_dir=CACHE_DATASETS,
                                            quantization_config=bnb_config,
                                            torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False) 

    # Adapter settings
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules = ["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    model.config.use_cache = False
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,cache_dir=CACHE_DATASETS)
#                                             torch_dtype=torch.bfloat16)

model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Training Parameters
training_bs = 16
evalutation_bs = 8
num_train_epochs = 7

## OPTIMIZER Parameters
weight_decay = 5e-3
learning_rate = 5e-4
lr_scheduler_type = "linear"
warmup_ratio = 0.2

training_args = TrainingArguments(
    report_to="wandb", # enables logging to W&B 😎
    run_name=RUN_NAME,
    per_device_train_batch_size=training_bs,
    per_device_eval_batch_size=evalutation_bs, 
    lr_scheduler_type=lr_scheduler_type,
    optim="adamw_torch" # "paged_adamw_32bit", # "adamw_torch",
    warmup_ratio=warmup_ratio,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    overwrite_output_dir='True',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=150,
    # save_total_limit = 1,
    # label_smoothing_factor=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    gradient_accumulation_steps=4, # simulate larger batch sizes
    # eval_accumulation_steps=100,
    output_dir=OUTPUT_DIR,
    bf16=True,
)

if EVALUATION_SAMPLES != -1:
    eval_subset_idx = sample(list(range(dataset_translation["validation"])), EVALUATION_SAMPLES)
else:
    eval_subset_idx = list(range(dataset_translation["validation"]))

trainer = SFTTrainer(
    model,
    train_dataset=dataset_translation["train"],
    eval_dataset=dataset_translation["validation"].select(eval_subset_idx),
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=1024,
    args=training_args,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(OUTPUT_DIR+f"/minerva_{MODEL_SIZE}_finetuned")

tokenizer.save_pretrained(OUTPUT_DIR+f"/minerva_{MODEL_SIZE}_finetuned")
