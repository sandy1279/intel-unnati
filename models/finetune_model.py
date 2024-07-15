import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

def formatted_train(input, response) -> str:
    return f"\n{input}</s>\n\n{response}</s>"

if __name__ == "__main__":
    dataset = "data/color_data"
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_model = "tinyllama-colorist-v1"

    data = Dataset.load_from_disk(dataset)
    model, tokenizer = get_model_and_tokenizer(model_id)

    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    training_arguments = TrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=3,
        max_steps=250,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024
    )

    trainer.train()
