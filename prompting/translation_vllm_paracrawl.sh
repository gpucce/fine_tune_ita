source /media/luca/minerva_sft_env/bin/activate

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "sapienzanlp/Minerva-3B-base-v1.0" \
    --dataset "paracrawl" \
    --from_lang it \
    --to_lang en

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "sapienzanlp/Minerva-3B-base-v1.0" \
    --dataset "paracrawl" \
    --from_lang en \
    --to_lang it

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "sapienzanlp/Minerva-1B-base-v1.0" \
    --dataset "paracrawl" \
    --from_lang it \
    --to_lang en

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "sapienzanlp/Minerva-1B-base-v1.0" \
    --dataset "paracrawl" \
    --from_lang en \
    --to_lang it

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "mistralai/Mistral-7B-v0.1" \
    --dataset "paracrawl" \
    --from_lang it \
    --to_lang en

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "mistralai/Mistral-7B-v0.1" \
    --dataset "paracrawl" \
    --from_lang en \
    --to_lang it

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "google/gemma-2b" \
    --dataset "paracrawl" \
    --from_lang it \
    --to_lang en

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "google/gemma-2b" \
    --dataset "paracrawl" \
    --from_lang en \
    --to_lang it

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "Qwen/Qwen2-1.5B" \
    --dataset "paracrawl" \
    --from_lang it \
    --to_lang en

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "Qwen/Qwen2-1.5B" \
    --dataset "paracrawl" \
    --from_lang en \
    --to_lang it

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "Qwen/Qwen2-7B" \
    --dataset "paracrawl" \
    --from_lang it \
    --to_lang en

python /home/luca/__Work/minerva_sft/prompting/translation_vllm.py --model_name "Qwen/Qwen2-7B" \
    --dataset "paracrawl" \
    --from_lang en \
    --to_lang it