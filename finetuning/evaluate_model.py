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
from accelerate import Accelerator
import json

os.environ['WANDB_MODE'] ="offline"

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




def call_model(examples, pipeline, tokenizer, BATCH_SIZE):
    texts = generate_prompt_examples(examples["source"])

    #outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    # outputs =  [model.generate(**tokenizer(
    #     i, return_tensors = "pt").to("cuda"), max_new_tokens = 128, use_cache = True, temperature=0.7, pad_token_id=tokenizer.unk_token_id) for i in texts]
    # outputs = model.generate(**tokenizer(texts, return_tensors = "pt", padding=True, max_length=1000, truncation=True).to("cuda"), max_new_tokens = 2, use_cache = True)
    # preds =  [o[0] for o in outputs]

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

    # outputs = tokenizer.batch_decode(outputs, skip_special_tokens = True)
    # outputs = [tokenizer.batch_decode(o, skip_special_tokens = True)[0] for o in outputs]

    return {"text": examples["source"], "outputs":outputs ,"labels": examples["target"]}


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

    SUBSET = 10
    SEED = 42

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 1

    random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_name)#, cache_dir=CACHE_DATASETS, revision=REVISION_ID)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [prompt_template, response_template]})
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
                                                    low_cpu_mem_usage=True,)
    labels = []
    decoded_preds = []
    preds = []
    SIZE = "350m"
    USE_SPLIT = "test"
    # TEXTGENERATIONPIPELINE
    pipeline = TextGenerationPipeline(
                    model=model, tokenizer=tokenizer
                )

    #model = model.to(DEVICE)
    pipeline.device = model.device

    print(f"DEVICE:\n\tSelected Device: {DEVICE}\n\tmodel Device: {model.device}\n\tpipeline Device: {pipeline.device}")

    # DATASET
    # News dataset is defined as union of Fanpage and IlPost
    dataset_fanpage = load_dataset("ARTeLab/fanpage")#, cache_dir=CACHE_DATASETS)
    dataset_ilpost = load_dataset("ARTeLab/ilpost")#, cache_dir=CACHE_DATASETS)
    # train the model over the training + validation sets 
    dataset_newsum = DatasetDict()
    dataset_newsum["train"] = concatenate_datasets([dataset_fanpage["train"], dataset_ilpost["train"]])
    dataset_newsum["validation"] = concatenate_datasets([dataset_fanpage["validation"], dataset_ilpost["validation"]])
    dataset_newsum["test"] = concatenate_datasets([dataset_fanpage["test"], dataset_ilpost["test"]])

    rouge = evaluate.load('rouge') ## import the rouge scorer
    if SUBSET == -1:
        val_sample = sample(list(range(len(dataset_newsum[USE_SPLIT]))), len(dataset_newsum[USE_SPLIT]))
    else:
        val_sample = sample(list(range(len(dataset_newsum[USE_SPLIT]))), SUBSET)

    texts = []

    for item in dataset_newsum[USE_SPLIT].select(val_sample).map(lambda x:call_model(x, pipeline, tokenizer, BATCH_SIZE), batched = True, batch_size = BATCH_SIZE, keep_in_memory=True, num_proc=1):
        decoded_preds.append(item["outputs"])
    #     preds.append(item["preds"])
        labels.append(item["labels"])
        texts.append(item["text"])

    predictions = model.predict()
    with open(f"predictions_{REVISION_ID}_{SIZE}.jsonl", "w") as f:
            for text, dec, lab in zip(texts, decoded_preds, labels):
                f.writelines(json.dumps({"text": text, "gold": lab, "predicted": dec}) + "\n")

    # Some simple post-processing
    decoded_preds_processed, decoded_labels = postprocess_text(decoded_preds, labels)

    result = rouge.compute(predictions=decoded_preds_processed, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #result["gen_len"] = np.mean(prediction_lens)

    print("="*30)
    print(f"REVISION: {REVISION_ID} - SPLIT: {USE_SPLIT}")
    print(result)
    print("="*30)

    #with open(f"results_{REVISION_ID}_{SIZE}_better_fit_test.jsonl", "w") as f:
    #    f.writelines(json.dumps(result) + "\n")
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Instruction evaluation',
                    description='...',
                    epilog='...')
    
    parser.add_argument('-c', '--config_path')      # option that takes a value

    args = parser.parse_args()

    main(args)