from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BertTokenizerFast
import torch
import json
import os

# 1. Załaduj model i tokenizer
model_dir = "/root/question_extraction/trained_model"
model = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)

def get_answer(model, tokenizer, question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    return answer

def evaluate_model(model, tokenizer, examples):
    correct = 0
    total = len(examples)
    
    for example in examples:
        context = example['context']
        question = example['question']
        predicted_answer = get_answer(model, tokenizer, question, context)
        
        print(f"Kontekst: {context}")        
        print(f"Pytanie z kontekstu: {question}")
        print(f"Odpowiedź modelu: {predicted_answer}")
        print("---")
        
    print(f"Liczba przykładów: {total}")

# Sprawdź, czy plik z przykładami testowymi istnieje
test_file = 'test_examples.jsonl'
if not os.path.exists(test_file):
    print(f"Plik {test_file} nie istnieje. Proszę upewnić się, że plik jest w odpowiednim miejscu.")
    exit(1)

# Wczytaj przykłady testowe
with open(test_file, 'r', encoding='utf-8') as f:
    test_examples = [json.loads(line) for line in f]

# Oceń model na przykładach testowych
evaluate_model(model, tokenizer, test_examples)