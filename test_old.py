from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import json
import os

# 1. Load the model and tokenizer
model_dir = "/root/question_extraction/trained_model_new"
model_checkpoints_dir = "/root/question_extraction/results_new/checkpoint-500"
model = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

def get_answer(model, tokenizer, question, context):
    # Tokenize input and generate model output
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract answer span from model output
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    print(f"Answer start: {answer_start}, Answer end: {answer_end}")
    
    if answer_end <= answer_start:
        return "Model nie mógł znaleźć odpowiedzi."
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    print(f"Raw answer: '{answer}'")
    
    return answer if answer.strip() else "Model nie mógł znaleźć odpowiedzi."

def evaluate_model(model, tokenizer, examples):
    model.eval()  # Dodaj tę linię tutaj
    correct = 0
    total = len(examples)
    
    for example in examples:
        context = example['context']
        question = example['question']
        predicted_answer = get_answer(model, tokenizer, question, context)
        
        # Print evaluation results for each example
        print(f"Kontekst: {context}")        
        print(f"Pytanie z kontekstu: {question}")
        print(f"Odpowiedź modelu: {predicted_answer}")
        print("---")
        
    print(f"Liczba przykładów: {total}")

# Check if the test examples file exists
test_file = 'test_examples.jsonl'
if not os.path.exists(test_file):
    print(f"Plik {test_file} nie istnieje. Proszę upewnić się, że plik jest w odpowiednim miejscu.")
    exit(1)

# Load test examples from JSONL file
with open(test_file, 'r', encoding='utf-8') as f:
    test_examples = [json.loads(line) for line in f]

# Evaluate the model on test examples
evaluate_model(model, tokenizer, test_examples)