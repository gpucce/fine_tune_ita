from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets, DatasetDict
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model  
import torch
import numpy as np
import evaluate
import nltk
import yaml
import argparse
from accelerate import Accelerator

def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

# FORMATTING
def generate_formatting_prompts_func(tokenizer, prompt_template, response_template):
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['source'])):
            text = f"{prompt_template} {example['source'][i]}\n{response_template} {example['target'][i]}{tokenizer.eos_token}"
            output_texts.append(text)
        return output_texts
    
    return formatting_prompts_func


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def generate_compute_metrics(tokenizer, metric_name="rouge"):
    # Metric
    metric = evaluate.load(metric_name)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        print(preds)
        print(labels)

        # TODO: debug -> tokens not related to the answer are setted as -100?

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
    
    return compute_metrics

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

    # DATASET
    print("## Load Dataset...")

    # News dataset is defined as union of Fanpage and IlPost
    dataset_fanpage = load_dataset("ARTeLab/fanpage")
    dataset_ilpost = load_dataset("ARTeLab/ilpost")
    # train the model over the training + validation sets 
    dataset_newsum = DatasetDict()
    dataset_newsum["train"] = concatenate_datasets([dataset_fanpage["train"], dataset_ilpost["train"]])
    dataset_newsum["validation"] = concatenate_datasets([dataset_fanpage["validation"], dataset_ilpost["validation"]])
    dataset_newsum["test"] = concatenate_datasets([dataset_fanpage["test"], dataset_ilpost["test"]])

    # TOKENIZER
    print("## Initialize Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    initial_token_count = len(tokenizer)
    added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [prompt_template, response_template]})

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
    

    model.save_pretrained("./shared_lora.config")

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = TrainingArguments(
        report_to="wandb", # enables logging to W&B 😎
        run_name=output_dir.split("/")[-1],
        per_device_train_batch_size=micro_training_bs,
        per_device_eval_batch_size=evaluation_bs, 
        lr_scheduler_type=lr_scheduler_type,
        optim="adamw_torch",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        overwrite_output_dir='True',
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=4, # TODO: define the correct value, looking at the total num of steps 
        eval_steps=25, # TODO: define the correct value, looking at the total num of steps 
        save_steps=512, # TODO: define the correct value, looking at the total num of steps 
        # save_total_limit = 1,
        # label_smoothing_factor=0.1,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        gradient_accumulation_steps=training_bs // micro_training_bs, # simulate larger batch sizes
        eval_accumulation_steps=16,
        output_dir=output_dir,
        bf16=True,
    )

    print("Per Device Micro Batch Size:", micro_training_bs)
    print("Actual Batch Size:", training_bs)

    import random

    random.seed(42)

    idx = random.sample(range(len(dataset_newsum["validation"])), 4096)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset_newsum["train"],
        eval_dataset=dataset_newsum["validation"].select(idx),
        tokenizer=tokenizer,
        formatting_func=generate_formatting_prompts_func(tokenizer, prompt_template, response_template),
        # compute_metrics=generate_compute_metrics(tokenizer, "rouge"),
        data_collator=collator,
        max_seq_length=2048,
        args=training_args
    )

    trainer.train()

    # trainer.save_model(output_dir)
    accelerator = trainer.accelerator
    
    unwrapped_model = trainer.model
    unwrapped_model = unwrapped_model.merge_and_unload()
    
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Instruction tuning',
                    description='...',
                    epilog='...')
    
    parser.add_argument('-c', '--config_path')      # option that takes a value

    args = parser.parse_args()

    main(args)