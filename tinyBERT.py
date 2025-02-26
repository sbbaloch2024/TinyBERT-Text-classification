import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import time
import evaluate  # Correct import for loading metrics

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Load TinyBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Rename label column to match model expectations
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set format for PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Select subsets for faster training (adjust as needed)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))  # For demonstration
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Load TinyBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    "huawei-noah/TinyBERT_General_4L_312D",
    num_labels=2
).to(device)

# Define training hyperparameters
learning_rate = 5e-5
batch_size = 16
epochs = 3
weight_decay = 0.01

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_tinybert",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=weight_decay,
    logging_dir="./logs_tinybert",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Load evaluation metric using the evaluate library
metric = evaluate.load("accuracy")

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
print("Starting fine-tuning TinyBERT...")
trainer.train()

# Evaluate the model
print("\nEvaluation Results for TinyBERT:")
eval_results = trainer.evaluate()
print(eval_results)

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_tinybert")
tokenizer.save_pretrained("./fine_tuned_tinybert")
