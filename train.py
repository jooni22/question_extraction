from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import TrainingArguments, Trainer
import torch
import json

# Wczytanie danych z pliku optimized_dataset.jsonl
with open("optimized_dataset.jsonl", "r") as file:
    data = [json.loads(line) for line in file]

# Przygotowanie danych treningowych
train_contexts = [item["context"] for item in data]
train_questions = [item["question"] for item in data]

# Inicjalizacja tokenizera i modelu BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Tokenizacja danych treningowych
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)

# Dodanie pozycji początku i końca odpowiedzi
train_encodings["start_positions"] = []
train_encodings["end_positions"] = []

for i in range(len(train_questions)):
    question = train_questions[i].lower().rstrip('?')  # Usuwamy znak zapytania, jeśli istnieje
    context = train_contexts[i].lower()
    
    # Znajdź pozycję pytania w kontekście
    start_position = context.find(question)
    if start_position != -1:
        end_position = start_position + len(question) - 1
    else:
        # Jeśli nie znaleziono dokładnego dopasowania, szukaj najbardziej podobnego fragmentu
        words = question.split()
        best_match = ""
        best_match_position = -1
        for j in range(len(words)):
            for k in range(j+1, len(words)+1):
                phrase = " ".join(words[j:k])
                pos = context.find(phrase)
                if pos != -1 and len(phrase) > len(best_match):
                    best_match = phrase
                    best_match_position = pos
        
        if best_match_position != -1:
            start_position = best_match_position
            end_position = start_position + len(best_match) - 1
        else:
            start_position = end_position = 0  # Jeśli nie znaleziono, ustaw na 0
    
    # Konwertuj pozycje znaków na pozycje tokenów
    tokenized_context = tokenizer.encode(train_contexts[i])
    start_token = len(tokenizer.encode(train_contexts[i][:start_position], add_special_tokens=False))
    end_token = len(tokenizer.encode(train_contexts[i][:end_position+1], add_special_tokens=False)) - 1
    
    train_encodings["start_positions"].append(start_token)
    train_encodings["end_positions"].append(end_token)

# Przygotowanie danych do treningu
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = QADataset(train_encodings)

# Konfiguracja parametrów treningu
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Inicjalizacja trenera
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Rozpoczęcie treningu
trainer.train()

# Zapisanie wytrenowanego modelu
model.save_pretrained("./trained_model")