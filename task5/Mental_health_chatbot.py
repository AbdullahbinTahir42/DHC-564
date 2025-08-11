
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 1. Define the model we want to use
model_name = "EleutherAI/gpt-neo-125M"

# 2. Load a reliable, alternative dataset
# The 'daily_dialog' dataset is well-maintained and won't cause loading errors.
print("Loading 'daily_dialog' dataset...")
raw_datasets = load_dataset("daily_dialog")

# 3. Load the tokenizer for our chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4. Create an updated preprocessing function for the new dataset's structure
# The 'daily_dialog' dataset has a 'dialog' column, which is a list of conversation turns.
def preprocess_function(examples):
    formatted_texts = []
    for dialog in examples['dialog']:
        # We need at least two turns to form a USER prompt and a BOT response.
        if len(dialog) > 1:
            # The first turn is the user, the second is the bot's empathetic reply.
            prompt = dialog[0]
            utterance = dialog[1]
            formatted_texts.append(f"USER: {prompt}\nBOT: {utterance}{tokenizer.eos_token}")
            
    return tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=128)

# 5. Apply the preprocessing function
# We process the dataset to create our prompt-response pairs.
print("Preprocessing dataset...")
tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets['train'].column_names # Discard original columns
)

# After mapping, the dataset only contains 'input_ids', 'attention_mask', etc.
# We need to re-split it into train and validation sets.
processed_data = tokenized_datasets["train"].train_test_split(test_size=0.1)


# 6. Load the pre-trained model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# 7. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
)

# 8. Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_data["train"],
    eval_dataset=processed_data["test"], # Use the new test split for evaluation
    tokenizer=tokenizer,
)

# 9. Start Fine-Tuning!
print("Starting training...")
trainer.train()

# 10. Save the fine-tuned model
print("Saving the fine-tuned model...")
trainer.save_model("./my_empathetic_chatbot")

print("\nTraining complete!")