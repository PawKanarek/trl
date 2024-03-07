accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml --num_processes=1 \
    examples/scripts/sft.py \
    --model_name google/gemma-7b \
    --dataset_name OpenAssistant/oasst_top1_2023-08-25 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --save_steps 20_000 \
    --use_peft \
    --lora_r 16 --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --load_in_4bit \
    --output_dir gemma-finetuned-openassistant

subtensor-mainnet-lite  | 2024-03-07 01:18:01 Error while running root epoch: "Not the block to update emission values."    


# mine

export LD_LIBRARY_PATH=/home/raix/miniconda3/envs/tpu/lib:$LD_LIBRARY_PATH

accelerate launch \
    examples/scripts/sft.py \
    --model_name google/gemma-7b \
    --dataset_name OpenAssistant/oasst_top1_2023-08-25 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --save_steps 20_000 \
    --output_dir gemma-finetuned-openassistant



accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml --num_processes=8 \
    examples/scripts/sft.py \
    --model_name google/gemma-7b \
    --dataset_name OpenAssistant/oasst_top1_2023-08-25 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --save_steps 20_000 \
    --use_peft \
    --lora_r 16 --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --output_dir gemma-finetuned-openassistant


python examples/scripts/sft.py \
    --model_name_or_path='google/gemma-7b' \
    --dataset_name='OpenAssistant/oasst_top1_2023-08-25' \
    --report_to="wandb" \
    --learning_rate=2e-4 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=1 \
    --output_dir="aayy-finetuned-openassistant" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=32 \
    --load_in_4bit \
    --lora_target_modules q_proj k_proj v_proj o_proj