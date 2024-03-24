import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
import torch
# Placeholder for tokenizer and model loading, adjust based on actual availability
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define custom configuration for LLaMA-2
custom_config = AutoConfig.from_pretrained(model_name,
                                           hidden_size=1024,
                                           num_attention_heads=32,
                                           num_hidden_layers=12,
                                           max_position_embeddings=4096,
                                           intermediate_size=4096)

# Initialize the model with the custom configuration
model = AutoModelForCausalLM.from_config(custom_config)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Check if the tokenized data directory exists
tokenized_data_dir = "./tokenized_data"
if os.path.exists(tokenized_data_dir):
    # Load the tokenized data from disk
    tokenized_datasets = load_from_disk(tokenized_data_dir)
else:
    # Load and preprocess the dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # Tokenize the texts and create labels by shifting the input_ids
        tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        # For language modeling, the labels are the input_ids shifted by one token
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # Save the tokenized data to disk for future use
    tokenized_datasets.save_to_disk(tokenized_data_dir)

# Define training arguments with max_steps set to 10
training_args = TrainingArguments(
    output_dir="./custom_llama2_results",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,  # Adjust based on your GPU capabilities
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    max_steps=10,  # Stop training after 10 batches
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Start training
trainer.train()
