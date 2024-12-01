import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Step 1: Load the model and tokenizer
def load_model_and_tokenizer(model_name="gpt2"):
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

# Step 2: Prepare dataset
def load_data(file_path=None, input_text=None):
    if file_path:
        # Load data from a file
        with open(file_path, 'r') as file:
            data = file.read()
    elif input_text:
        # Use the input text
        data = input_text
    else:
        raise ValueError("No data provided. Provide either a file path or input text.")

    # Create a dataset
    data = [data]  # Wrap in list to match expected dataset structure
    return Dataset.from_dict({"text": data})

# Step 3: Tokenize data
def tokenize_data(dataset, tokenizer):
    print("Tokenizing data...")
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
    
    return dataset.map(tokenize_function, batched=True)

# Step 4: Fine-tune the model
def fine_tune_model(model, dataset, output_dir="output_model", epochs=3, batch_size=4):
    print("Starting fine-tuning...")
    training_args = TrainingArguments(
        output_dir=output_dir,  # Where to save the model after training
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=200,
        # Ensure the CPU is used (avoid using CUDA/GPU)
        no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# Main function to execute the workflow
def main(data_file=None, input_text=None):
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_data(file_path=data_file, input_text=input_text)
    dataset = tokenize_data(dataset, tokenizer)
    
    # Ensure the model runs on CPU explicitly
    model.to(torch.device('cpu'))
    
    # Fine-tune the model
    fine_tune_model(model, dataset)

if __name__ == "__main__":
    # Provide your data source here: a file path or direct input text
    data_file = "data.txt"  # Replace with your file path or None
    input_text = None  # Replace with some text if you prefer entering it directly
    
    main(data_file=data_file, input_text=input_text)
