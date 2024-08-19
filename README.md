# Question Answering Model Evaluation

This project evaluates a pre-trained Question Answering model using the Transformers library from Hugging Face.

## Project Description

A Question Answering model is an advanced artificial intelligence tool capable of understanding context and answering questions based on it. This project focuses on evaluating such a model, allowing us to assess its effectiveness and accuracy in various scenarios.

## Requirements

To run this project, you need:

- Python 3.6 or newer
- PyTorch
- Transformers library
- JSON

## Environment Setup

1. First, make sure you have Python 3.6 or newer installed. You can check this by typing in the terminal:
   ```
   python --version
   ```

2. Then install the required libraries. You can do this using pip:
   ```
   pip install torch transformers
   ```

3. Place the pre-trained model in the `/root/question_extraction/trained_model_new` directory. Ensure you have the appropriate permissions for this directory.

4. Prepare a JSONL file named `test_examples.jsonl` with test examples. Each line should contain one example in the following format:
   ```json
   {"context": "Example context", "question": "Example question"}
   ```

## Project Structure

The project consists of the following main elements:

- `test.py`: Main script for model evaluation.
- `test_examples.jsonl`: JSONL file containing test examples.
- `/root/question_extraction/trained_model_new/`: Directory containing pre-trained model files.
- `README.md`: This file, containing project documentation.

## Usage

To run the model evaluation, follow these steps:

1. Open a terminal and navigate to the project directory.

2. Ensure that the `test_examples.jsonl` file is in the same directory as the `test.py` script.

3. Run the script using the command:
   ```
   python test.py
   ```

4. The script will perform the following operations:
   - Load the pre-trained model and tokenizer.
   - Read test examples from the `test_examples.jsonl` file.
   - Evaluate the model on each example.
   - Print the context, question, and model's answer for each example.
   - Display the total number of processed examples.

## Results

After the evaluation is complete, you will see a result for each example in the following format:

```
Context: [context content]
Question from context: [question content]
Model's answer: [answer generated by the model]
---
```

At the end, the total number of processed examples will be displayed.

## Troubleshooting

- If you encounter an error related to the missing `test_examples.jsonl` file, make sure this file is in the same directory as the `test.py` script.
- In case of problems with loading the model, check if the path `/root/question_extraction/trained_model_new` is correct and if you have the appropriate access permissions.

## Further Development

This project can be extended in many ways, for example:

- Adding metrics to evaluate the quality of the model's answers.
- Implementing a user interface for easier input of questions and contexts.
- Extending functionality to include fine-tuning the model on custom data.

## License

This project is released under the MIT License. See the LICENSE file for details.

## Contact

For questions or suggestions, please contact [your_email@example.com].