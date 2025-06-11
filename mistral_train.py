from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

model_name = "mistralai/Mistral-7B-v0.1"  # Sau model local dacă l-ai descărcat

# Incarcă tokenizer și model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # necesar pt padding

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

# Pregătește modelul pentru LoRA
model = prepare_model_for_kbit_training(model)

# Config LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Încarcă datasetul tău
dataset = load_dataset("json", data_files="instruction_dataset.jsonl")["train"]

# Preprocesare text (în format prompt-style)
def format_prompt(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"input_ids": tokenizer(prompt, truncation=True, padding="max_length", max_length=512)["input_ids"]}

dataset = dataset.map(format_prompt, batched=False)

# Config trainer
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    push_to_hub=False
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,
)

trainer.train()
