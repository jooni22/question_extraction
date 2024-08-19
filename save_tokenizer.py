from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained("/root/question_extraction/trained_model/")
print("Tokenizer saved successfully.")