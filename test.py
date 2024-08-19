import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Wczytaj model i tokenizer
model_path = "./trained_model_new/"
model_checkpoint_path = "./results_new/checkpoint-1500/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint_path)

# Funkcja do znajdowania odpowiedzi w kontekście
def find_answer(context, question):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    # Rozszerz zakres odpowiedzi, aby uwzględnić znak zapytania
    while answer_end < len(inputs["input_ids"][0]) and inputs["input_ids"][0][answer_end] != tokenizer.sep_token_id:
        if tokenizer.decode(inputs["input_ids"][0][answer_end]) == "?":
            answer_end += 1
            break
        answer_end += 1
    
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
    return answer.strip()

# Wczytaj dane testowe
with open("test_examples.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

# Przeprowadź ewaluację
correct = 0
total = len(test_data)

for example in test_data:
    context = example["context"]
    true_question = example["question"]
    
    # Znajdź pytanie w kontekście
    predicted_question = find_answer(context, "What is the question in this context?")
    
    # Porównaj przewidziane pytanie z prawdziwym
    if predicted_question.lower() == true_question.lower():
        correct += 1
    
    # Wypisz wyniki dla każdego przykładu
    print(f"Context: {context[:100]}...")
    print(f"True question: {true_question}")
    print(f"Predicted question: {predicted_question}")
    print("-" * 50)

# Oblicz i wypisz dokładność
accuracy = correct / total
print(f"Accuracy: {accuracy:.2f}")