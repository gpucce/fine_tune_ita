import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextGenerationPipeline
from datasets import load_dataset
from random import sample
from tqdm import tqdm

# VARS
CACHE_DATASETS="/media/luca/hf_cache"
LANG_MAP = {
    "it": "italian",
    "en": "english"
}
model_name = "sapienzanlp/Minerva-3B-base-v1.0"
dataset = "flores"
from_lang = "it"
to_lang = "en"

# Definition of evaluation metric
comet_metric = evaluate.load('comet') ## import the rouge scorer
bleu_metric = evaluate.load('bleu')
# ----

def create_few_shot_example(text, train_data, from_lang, to_lang, num_shots=5):
    prompt = ""

    # random pick few shot samples from training data
    few_shots_subset_idx = sample(list(range(len(train_data))), num_shots)

    for idx in few_shots_subset_idx:
        prompt += f"translate {LANG_MAP[from_lang]} to {LANG_MAP[to_lang]}\n###Text: {train_data[from_lang][idx]}\n###Translation: {train_data[to_lang][idx]}\n\n"

    return prompt + f"translate {LANG_MAP[from_lang]} to {LANG_MAP[to_lang]}\n###Text: {text}\n###Translation: "


def formatting_prompts_func(examples, train_data, from_lang):
    output_texts = []
    for i in tqdm(range(len(examples["en"]))):

        text = create_few_shot_example(examples[from_lang][i], train_data,from_lang, to_lang, 5)

        output_texts.append(text)
    return output_texts


def main():
    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(device)

    # model, tokenizer, and pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DATASETS)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DATASETS, torch_dtype=torch.bfloat16, device_map="cuda:0")
    
    pipeline = TextGenerationPipeline(
                model=model, tokenizer=tokenizer
            )

    pipeline.device = model.device
    pipeline.tokenizer.pad_token_id = model.config.eos_token_id

    # load_dataset
    train_data = load_dataset("csv", data_files=f"/home/luca/__Work/minerva_sft/Flores/flores.{from_lang}-{to_lang}/dev.csv", index_col=False, cache_dir=CACHE_DATASETS)["train"]
    train_data = train_data.rename_column("src", from_lang)
    train_data = train_data.rename_column("ref", to_lang)

    test_data = load_dataset("csv", data_files=f"/home/luca/__Work/minerva_sft/Flores/flores.{from_lang}-{to_lang}/test.csv", index_col=False, cache_dir=CACHE_DATASETS)["train"]
    test_data = test_data.rename_column("src", from_lang)
    test_data = test_data.rename_column("ref", to_lang)
    
    texts = formatting_prompts_func(test_data, train_data, from_lang)

    sources = [test_data[from_lang][i] for i in range(len(test_data["en"]))]
    references = [test_data[to_lang][i] for i in range(len(test_data["en"]))]

    predicted = [
            out[0]["generated_text"]  # type: ignore
            for out in pipeline(
                texts,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                max_new_tokens=100,
                batch_size=1,
                num_beams=1,
            )  # type: ignore
        ]

    predicted = [p.split("\n\n")[0] for p in predicted]

    comet_result = comet_metric.compute(predictions=predicted, references=references, sources=sources)
    bleu_result = bleu_metric.compute(predictions=predicted, references=references)
    result = {"comet": round(result["mean_score"] * 100, 4), "bleu": bleu_result["bleu"]}

    print(result)


if __name__ == "__main__":
    main()