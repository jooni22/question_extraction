import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from tqdm import tqdm

# Klasa do wczytywania zoptymalizowanego datasetu
class OptimizedDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Funkcja do tokenizacji danych
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["context"],
        examples["question"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

# Funkcja do trenowania modelu
def train_model(model, train_dataloader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average loss: {avg_loss:.4f}")

# Główna funkcja
def main():
    # Wczytaj zoptymalizowany dataset
    dataset = OptimizedDataset("optimized_dataset.jsonl")
    
    # Inicjalizacja tokenizera i modelu
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    
    # Tokenizacja datasetu
    tokenized_dataset = [tokenize_function(example, tokenizer) for example in dataset]
    
    # Przygotowanie dataloadera
    batch_size = 16
    train_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    
    # Konfiguracja optymalizatora
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Sprawdzenie dostępności GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Trenowanie modelu
    num_epochs = 3
    train_model(model, train_dataloader, optimizer, device, num_epochs)
    
    # Zapisanie wytrenowanego modelu
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    print("Model został zapisany.")

if __name__ == "__main__":
    main()
