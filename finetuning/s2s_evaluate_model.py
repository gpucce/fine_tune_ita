from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Text2TextGenerationPipeline
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

from transformers import pipelines



SIZE = "350m"
USE_SPLIT = "test"
CACHE_DATASETS="/media/luca/hf_cache"
# OUTPUT_DIR=f"/media/luca/minerva_summary/news_minerva_lora_{SIZE}_eval"
MODEL_NAME =  str(sys.argv[1]) # "iperbole/news_minerva_no_smoothing_3b" # | first parameter is the model name

SUBSET = 10
SEED = 42

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1

random.seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DATASETS)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = 'left'

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,cache_dir=CACHE_DATASETS,
                                            torch_dtype=torch.bfloat16)

# TEXTGENERATIONPIPELINE
#pipeline = Text2TextGenerationPipeline(
#                model=model, tokenizer=tokenizer
#            )

model = pipelines.pipeline("summarization", model='ARTeLab/mbart-summarization-fanpage')

# model = model.to(DEVICE)
#pipeline.device = model.device

#print(f"DEVICE:\n\tSelected Device: {DEVICE}\n\tmodel Device: {model.device}\n\tpipeline Device: {pipeline.device}")

# DATASET
# News dataset is defined as union of Fanpage and IlPost
dataset_fanpage = load_dataset("ARTeLab/fanpage", cache_dir=CACHE_DATASETS)
dataset_ilpost = load_dataset("ARTeLab/ilpost", cache_dir=CACHE_DATASETS)
# train the model over the training + validation sets 
dataset_newsum = DatasetDict()
dataset_newsum["train"] = concatenate_datasets([dataset_fanpage["train"], dataset_ilpost["train"]])
dataset_newsum["validation"] = concatenate_datasets([dataset_fanpage["validation"], dataset_ilpost["validation"]])
dataset_newsum["test"] = concatenate_datasets([dataset_fanpage["test"], dataset_ilpost["test"]])

rouge = evaluate.load('rouge') ## import the rouge scorer

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

labels = []
decoded_preds = []
preds = []

def call_model(examples):
    texts = examples["source"]

    print(model(texts))

    outputs = [
            out["summary_text"]  # type: ignore
            for out in model(texts)
            #for out in pipeline(
            #    texts,
            #    # return_full_text=False,
            #    clean_up_tokenization_spaces=True,
            #    max_new_tokens=128,
            #    # pad_token_id=tokenizer.unk_token_id,
            #    batch_size=1,
            #    num_beams=1,
            #)  # type: ignore
        ]

    return {"text": examples["source"], "outputs":outputs ,"labels": examples["target"]}

if SUBSET == -1:
    val_sample = sample(list(range(len(dataset_newsum[USE_SPLIT]))), len(dataset_newsum[USE_SPLIT]))
else:
    val_sample = sample(list(range(len(dataset_newsum[USE_SPLIT]))), SUBSET)

texts = []

for item in dataset_newsum[USE_SPLIT].select(val_sample).map(call_model, batched = True, batch_size = BATCH_SIZE, keep_in_memory=True, num_proc=1):
    decoded_preds.append(item["outputs"])
#     preds.append(item["preds"])
    labels.append(item["labels"])
    texts.append(item["text"])

import json

if SUBSET != -1:
    with open(f"predictions_mbart_fanpage.jsonl", "w") as f:
        for text, dec, lab in zip(texts, decoded_preds, labels):
            f.writelines(json.dumps({"text": text, "gold": lab, "predicted": dec}) + "\n")

# Some simple post-processing
decoded_preds_processed, decoded_labels = postprocess_text(decoded_preds, labels)

result = rouge.compute(predictions=decoded_preds_processed, references=decoded_labels, use_stemmer=True)
result = {k: round(v * 100, 4) for k, v in result.items()}
#prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#result["gen_len"] = np.mean(prediction_lens)

print("="*30)
print(f"SPLIT: {USE_SPLIT}")
print(result)
print("="*30)

# with open(f"results_s2s_{SIZE}.jsonl", "w") as f:
#     f.writelines(json.dumps(result) + "\n")