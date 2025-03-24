import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

import re

# 1. Configuration
BASE_MODEL = "gpt2"
ds_name = "StephanAkkerman/crypto-stock-tweets"

# 2. Optimized Dataset Loading
dataset = load_dataset(ds_name, split="train[:30%]")  # Reduced dataset size
dataset = dataset.filter(lambda x: len(x["text"]) > 50)  # Filter short tweets
dataset = dataset.remove_columns("url")

# 3. Efficient Text Processing
def clean_tweet(tweet):
    tweet = re.sub(r'#(\w+)', r'\1', tweet)  # Remove hashtag symbol
    tweet = re.sub(r'https?://\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'[^\w\s$%@.,!?&/-]', '', tweet)  # Clean special chars
    return re.sub(r'\s+', ' ', tweet).strip()[:280]  # Limit to tweet length

dataset = dataset.map(
    lambda x: {"text": [f"Crypto: {clean_tweet(txt)}" for txt in x["text"]]},
    batched=True,
    batch_size=1000,
    num_proc=4  # Parallel processing
)

# 4. Optimized Model Loading
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,  # Mixed precision
    device_map="auto"  # Automatic device placement
)
tokenizer.pad_token = tokenizer.eos_token

# 5. Streamlined Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=96,  # Reduced length
        padding="max_length",
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=4,
    remove_columns=["text"]
)

# 6. Optimized Training Parameters
training_args = TrainingArguments(
    output_dir="crypto_gpt2",
    num_train_epochs=1,
    per_device_train_batch_size=16,  # Increased batch size
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    fp16=True,  # Keep FP16 enabled
    gradient_checkpointing=False,  # Disabled to avoid error
    optim="adafactor",  # Memory-efficient optimizer
    logging_steps=50,
    save_total_limit=1,
    max_steps=2000,  # Hard limit steps
    report_to="none"
)

# 7. Training Setup
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8  # Better GPU utilization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 8. Start Training
trainer.train()