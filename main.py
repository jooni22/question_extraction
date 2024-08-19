import json
import random
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizer

# Inicjalizacja tokenizera
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def truncate_context(context, max_length=400):
    # Tokenizuj kontekst
    tokens = tokenizer.tokenize(context)
    
    # Jeśli kontekst jest krótszy niż max_length, zwróć go bez zmian
    if len(tokens) <= max_length:
        return context
    
    # Znajdź ostatnią kropkę w dozwolonym zakresie
    truncated_tokens = tokens[:max_length]
    last_period_index = len(truncated_tokens) - 1
    while last_period_index > 0 and truncated_tokens[last_period_index] != '.':
        last_period_index -= 1
    
    # Jeśli nie znaleziono kropki, użyj całego dozwolonego zakresu
    if last_period_index == 0:
        last_period_index = len(truncated_tokens) - 1
    
    # Połącz tokeny z powrotem w tekst
    truncated_context = tokenizer.convert_tokens_to_string(truncated_tokens[:last_period_index+1])
    return truncated_context

def refactor_data(example):
    context = truncate_context(example['context'])
    instruction = example['instruction']
    
    # Funkcja do sprawdzania i przycinania całkowitej długości
    def check_and_truncate(text, max_length=512):
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length:
            return tokenizer.convert_tokens_to_string(tokens[:max_length])
        return text
    
    # Podziel kontekst na zdania
    sentences = [s.strip() for s in context.split('.') if s.strip()]
    
    # Pierwsza wersja (oryginalna)
    if len(sentences) <= 1:
        new_context_1 = f"{context} {instruction}"
    else:
        insert_index = random.randint(1, len(sentences))
        sentences_1 = sentences.copy()
        sentences_1.insert(insert_index, instruction)
        new_context_1 = '. '.join(sentences_1).strip()
    
    new_context_1 = check_and_truncate(new_context_1)
    
    # Druga wersja (zmodyfikowana)
    instruction_without_question_mark = instruction.rstrip('?')
    if len(sentences) <= 1:
        new_context_2 = f"{instruction_without_question_mark}. {context}"
    else:
        insert_index = random.randint(1, len(sentences))
        sentences_2 = sentences.copy()
        sentences_2.insert(insert_index, instruction_without_question_mark)
        new_context_2 = '. '.join(sentences_2).strip()
    
    new_context_2 = check_and_truncate(new_context_2)
    
    return [
        {
            "instruction": example['instruction'],
            "context": new_context_1,
            "response": example['response'],
            "category": example['category']
        },
        {
            "instruction": instruction_without_question_mark,
            "context": new_context_2,
            "response": example['response'],
            "category": example['category']
        }
    ]

def save_dataset(dataset, filename):
    if os.path.exists(filename):
        print(f"Plik {filename} już istnieje. Pomijam zapisywanie.")
        return
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc=f"Zapisywanie {filename}", unit="example"):
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"Zapisano dataset do pliku {filename}")
    except Exception as e:
        print(f"Wystąpił błąd podczas zapisywania do pliku {filename}: {str(e)}")

def optimize_dataset(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Plik {input_file} nie istnieje. Nie można zoptymalizować.")
        return False
    
    if os.path.exists(output_file):
        print(f"Plik {output_file} już istnieje. Pomijam optymalizację.")
        return True
    
    try:
        total_sequences = 0
        filtered_sequences = 0
        with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc=f"Optymalizacja {output_file}", unit="line"):
                total_sequences += 1
                item = json.loads(line.strip())
                context_tokens = tokenizer.tokenize(item["context"])
                question_tokens = tokenizer.tokenize(item["instruction"])
                
                if len(context_tokens) + len(question_tokens) <= 512:
                    optimized_item = {"context": item["context"], "question": item["instruction"]}
                    json.dump(optimized_item, f_out, ensure_ascii=False)
                    f_out.write('\n')
                else:
                    filtered_sequences += 1
        
        print(f"Zoptymalizowany dataset zapisano do pliku {output_file}")
        print(f"Usunięto {filtered_sequences} sekwencji przekraczających 512 tokenów.")
        print(f"Pozostało {total_sequences - filtered_sequences} sekwencji.")
        return True
    except Exception as e:
        print(f"Wystąpił błąd podczas optymalizacji datasetu: {str(e)}")
        return False

def check_sequence_lengths(filename):
    long_sequences = 0
    total_sequences = 0
    max_length = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Sprawdzanie długości sekwencji"):
            item = json.loads(line)
            tokens = tokenizer.tokenize(item['context'] + ' ' + item['question'])
            length = len(tokens)
            total_sequences += 1
            max_length = max(max_length, length)
            if length > 512:
                long_sequences += 1
    
    print(f"Liczba sekwencji dłuższych niż 512 tokenów: {long_sequences}")
    print(f"Całkowita liczba sekwencji: {total_sequences}")
    print(f"Procent długich sekwencji: {long_sequences/total_sequences*100:.2f}%")
    print(f"Maksymalna długość sekwencji: {max_length}")

# Załaduj dataset
print("Ładowanie datasetu...")
try:
    dataset = load_dataset("databricks/databricks-dolly-15k")
except Exception as e:
    print(f"Wystąpił błąd podczas ładowania datasetu: {str(e)}")
    exit(1)

# Filtruj dataset, aby zawierał tylko przykłady z niepustym kontekstem
print("Filtrowanie datasetu...")
filtered_dataset = dataset['train'].filter(lambda example: example['context'].strip() != '')

# Refaktoryzacja danych
print("Refaktoryzacja danych...")
try:
    refactored_dataset = []
    for example in tqdm(filtered_dataset, desc="Refaktoryzacja", unit="example"):
        refactored_dataset.extend(refactor_data(example))
except Exception as e:
    print(f"Wystąpił błąd podczas refaktoryzacji danych: {str(e)}")
    exit(1)

# Zapisz oryginalny dataset (tylko z niepustymi kontekstami)
save_dataset(filtered_dataset, "original_dataset.jsonl")

# Zapisz przetworzony dataset
save_dataset(refactored_dataset, "refactored_dataset.jsonl")

# Optymalizuj dataset
optimization_success = optimize_dataset("refactored_dataset.jsonl", "optimized_dataset.jsonl")

# Wyświetl przykłady danych
def print_example(filename, description):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"\nPrzykład {description}:")
            print(json.dumps(json.loads(first_line), indent=2))
    except Exception as e:
        print(f"Nie udało się wyświetlić przykładu z {filename}: {str(e)}")

print_example("original_dataset.jsonl", "oryginalnych danych")
print_example("refactored_dataset.jsonl", "przetworzonych danych")
if optimization_success:
    print_example("optimized_dataset.jsonl", "zoptymalizowanych danych")

print(f"\nLiczba przykładów w oryginalnym datasecie: {len(dataset['train'])}")
print(f"Liczba przykładów w przefiltrowanym datasecie: {len(filtered_dataset)}")
print(f"Liczba przykładów w przetworzonym datasecie: {len(refactored_dataset)}")

# Sprawdź, czy pliki istnieją
print("\nSprawdzanie istnienia plików:")
for file in ["original_dataset.jsonl", "refactored_dataset.jsonl", "optimized_dataset.jsonl"]:
    if os.path.exists(file):
        print(f"Plik {file} istnieje.")
    else:
        print(f"Plik {file} nie istnieje.")

if optimization_success:
    check_sequence_lengths("optimized_dataset.jsonl")