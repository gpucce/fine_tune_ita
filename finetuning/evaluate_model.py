from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, BitsAndBytesConfig
import torch
import evaluate 
import pandas as pd
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm
import numpy as np
from random import sample
import nltk
import sys
import random
import os
import argparse
import yaml
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model  
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from accelerate import Accelerator
import json

os.environ['WANDB_MODE'] ="offline"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def postprocess_text(preds, labels, consider_just_n_sentence=1):
    """new_preds = []
    for pred in preds:
        if response_template in pred:
            responses = pred.strip().split(response_template)
            responses = [r.strip() for r in responses if len(r.strip()) > 0]
            new_preds.append(responses[1])
        else:
            new_preds.append("-")
    preds = new_preds
    """
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def generate_prompt_examples(texts):
    output_texts = []
    for txt in texts:
        text = f"### Text: {txt}\n ### Summary:"
        output_texts.append(text)
    return output_texts

def call_model(tokenizer, model, prompt_template, example, max_source_len):
    #texts = generate_prompt_examples(examples["source"])

    #outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    # outputs =  [model.generate(**tokenizer(
    #     i, return_tensors = "pt").to("cuda"), max_new_tokens = 128, use_cache = True, temperature=0.7, pad_token_id=tokenizer.unk_token_id) for i in texts]
    # outputs = model.generate(**tokenizer(texts, return_tensors = "pt", padding=True, max_length=1000, truncation=True).to("cuda"), max_new_tokens = 2, use_cache = True)
    # preds =  [o[0] for o in outputs]
    """
    outputs = [
            out[0]["generated_text"]  # type: ignore
            for out in pipeline(
                texts,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                max_new_tokens=128,
                pad_token_id=tokenizer.unk_token_id,
                batch_size=BATCH_SIZE,
                num_beams=1,
            )  # type: ignore
        ]
    """
    pred_output = []
    labels_output = []
    for i in range(len(example)):
        prompt = f"{prompt_template}{example['source'][i][:max_source_len]}"
        inputs = tokenizer(prompt, return_tensors="pt")
        generate_ids = model.generate(inputs.input_ids, max_length=max_source_len)
        pred = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred_output.append(pred)
        labels_output.append(example['target'][i])
        break
    # outputs = tokenizer.batch_decode(outputs, skip_special_tokens = True)
    # outputs = [tokenizer.batch_decode(o, skip_special_tokens = True)[0] for o in outputs]

    return pred_output, labels_output#{"text": examples["source"], "outputs":outputs ,"labels": examples["target"]}

def evaluate_model(preds, labels, tokenizer, metric_name="rouge"):
    
        
        #labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        #labels_output.append(labels)

    metric = evaluate.load(metric_name)
    result = metric.compute(predictions=preds, references=labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

def main(args):
    config_path = args.config_path

    # Read config YAML file
    with open(config_path, 'r') as f:
        config_loaded = yaml.safe_load(f)

    model_name = config_loaded["model_name"]
    use_lora = config_loaded["use_lora"]
    output_dir = config_loaded["output_dir"]
    response_template = config_loaded["response_template"]
    prompt_template = config_loaded["prompt_template"]
    micro_training_bs = config_loaded["micro_training_bs"]
    training_bs = config_loaded["training_bs"]
    evaluation_bs = config_loaded["evaluation_bs"]
    num_train_epochs = config_loaded["num_train_epochs"]
    weight_decay = config_loaded["weight_decay"]
    learning_rate = config_loaded["learning_rate"]
    lr_scheduler_type = config_loaded["lr_scheduler_type"]
    warmup_ratio = config_loaded["warmup_ratio"]
    max_source_len = config_loaded["max_source_len"]
    max_target_len = config_loaded["max_target_len"]
    max_seq_length = max_source_len + max_target_len
    results_dir = config_loaded["results_dir"]
    # DATASET
    print("## Load Dataset...")

    # News dataset is defined as union of Fanpage and IlPost
    dataset_fanpage = load_dataset("ARTeLab/fanpage")
    dataset_ilpost = load_dataset("ARTeLab/ilpost")
    # train the model over the training + validation sets 
    dataset_newsum = DatasetDict()
    #dataset_newsum["train"] = concatenate_datasets([dataset_fanpage["train"], dataset_ilpost["train"]])
    #dataset_newsum["validation"] = concatenate_datasets([dataset_fanpage["validation"], dataset_ilpost["validation"]])
    dataset_newsum["test"] = concatenate_datasets([dataset_fanpage["test"], dataset_ilpost["test"]])
    #dataset_newsum["train"] = dataset_newsum["train"]#.select(range(100))
    #dataset_newsum["validation"] = dataset_newsum["validation"]#.select(range(5000))
    dataset_newsum["test"] = dataset_newsum["test"].select(range(1))
    # TOKENIZER
    print("## Initialize Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    initial_token_count = len(tokenizer)
    added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [prompt_template, response_template]})
    print("initial_token_count = len(tokenizer) ", initial_token_count)
    #print("initial_token_count = len(tokenizer) +added_token_count", initial_token_count+added_token_count)
    # MODEL
    print("## Load Model...")

    if use_lora: ## LORA:
        ## LORA PARAMTERS
        r = config_loaded["lora_r"]
        lora_alpha = config_loaded["lora_alpha"]
        target_modules = config_loaded["target_modules"]
        lora_dropout = config_loaded["lora_dropout"]

        ## QUANTIZATION PARAMATERS
        compute_dtype = torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage="bfloat16",
        ).to_dict()

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                attn_implementation="flash_attention_2",
                                                torch_dtype=torch.bfloat16,
                                                device_map={"": get_current_device()} if torch.cuda.is_available() else None,
                                                quantization_config=quantization_config,
                                                low_cpu_mem_usage=True,
                                                )
        #model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False) 

        model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)

        # Adapter settings
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules = target_modules,
            lora_dropout=lora_dropout, 
            bias="none",
            task_type="CAUSAL_LM",
        )


        model = get_peft_model(model, lora_config)

        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.bfloat16,
                                                attn_implementation="flash_attention_2")
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    pred_output, labels_output = call_model(tokenizer, model, prompt_template, dataset_newsum["test"], max_source_len)
    preds, labels = postprocess_text(pred_output, labels_output)
    #preds = ['hello']
    #labels = ['hello']
    results = evaluate_model(preds, labels, tokenizer)
    with open(results_dir+"/results_better_fit_test.json", "w") as f:
        f.writelines(json.dumps(results) + "\n")
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Instruction evaluation',
                    description='...',
                    epilog='...')
    
    parser.add_argument('-c', '--config_path')      # option that takes a value

    args = parser.parse_args()

    main(args)