from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json
import torch

# 🧾 Încarcă datasetul (JSONL Alpaca-style)
def load_dataset_from_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            prompt = f"<s>[INST] {ex['instruction'].strip()} [/INST] {ex['output'].strip()} </s>"
            data.append({"text": prompt})
    return Dataset.from_list(data).train_test_split(test_size=0.1)

dataset = load_dataset_from_jsonl("instruction_dataset.jsonl")

# 🔤 Tokenizer + model (cu 4-bit)
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # important pentru padding

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# 🔧 Adaugă LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # specific pentru Mistral/LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 🧠 Tokenizare
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# 📦 Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ⚙️ Argumente antrenare
training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    push_to_hub=False
)

# 🏋️ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ▶️ Start fine-tuning
trainer.train()

# 💾 Salvează modelul final
trainer.save_model("./mistral-finetuned")
tokenizer.save_pretrained("./mistral-finetuned")
