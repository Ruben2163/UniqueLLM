from transformers import GPTJForCausalLM, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPTJForCausalLM.from_pretrained("./fine_tuned_gpt_j")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt_j")

# Set the pad token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Generate text based on a prompt
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)

print(tokenizer.decode(output[0], skip_special_tokens=True))
