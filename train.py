from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json
import torch

# 🧾 Load JSONL Alpaca-style dataset
def load_dataset_from_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            prompt = f"<s>[INST] {ex['instruction'].strip()} [/INST] {ex['output'].strip()} </s>"
            data.append({"text": prompt})
    return Dataset.from_list(data).train_test_split(test_size=0.1)

dataset = load_dataset_from_jsonl("instruction_dataset.jsonl")

# 📦 Model & Tokenizer: TinyLlama
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if torch.cuda.is_available():
    print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU is NOT available. Using CPU instead.")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,  # Requires bitsandbytes
    device_map="auto"
)

# 🔧 LoRA config (adjusted for TinyLlama, which uses q_proj/v_proj like LLaMA)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 🧠 Tokenization
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512  # TinyLlama context is ~2048, but keep light for 1.1B
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# 📦 Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ⚙️ Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=4,  # lighter model = bigger batch
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs",
    learning_rate=2e-5,
    num_train_epochs=2000,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    push_to_hub=False
)

# 🏋️ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ▶️ Fine-tune
trainer.train()

# 💾 Save model and tokenizer
trainer.save_model("./tinyllama-finetuned")
tokenizer.save_pretrained("./tinyllama-finetuned")
