import json
import random
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

# Sprawdź dostępność GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Załadowanie przetworzonych danych
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

data = load_jsonl("refactored_dataset.jsonl")
dataset = Dataset.from_list(data)

# 2. Przygotowanie modelu NLP
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name).to(device)

# 3. Przygotowanie danych treningowych
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    inputs["start_positions"] = []
    inputs["end_positions"] = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        question = questions[sample_idx]
        context = examples["context"][sample_idx]

        start_char = context.lower().index(question.lower())
        end_char = start_char + len(question)

        sequence_ids = inputs.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            inputs["start_positions"].append(0)
            inputs["end_positions"].append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            inputs["start_positions"].append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            inputs["end_positions"].append(idx + 1)

    return inputs

# Podziel dane na zbiór treningowy i walidacyjny
train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

# Tokenizacja i przygotowanie danych
tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
tokenized_val = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)

# 4. Trening modelu
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

# 5. Ewaluacja modelu
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 6. Zapisanie modelu
print("Saving model...")
model.save_pretrained("./qa_model")
tokenizer.save_pretrained("./qa_model")
print("Model saved successfully.")
