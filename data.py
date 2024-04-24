from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("csv", data_files="rephrased_questions.csv")["train"]

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-base")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

task_prefix = "answer"
def tokenize_function(examples):
    input_text = ["question: " + q + " answer: " + a for q,
                  a in zip(examples['Question'], examples['Answer'])]
    model_inputs = tokenizer(input_text, max_length=512, truncation=True,
                             padding="max_length", add_special_tokens=True)

    # Создание labels, которые начинаются с токена начала строки, и установка значения -100 для padding
    labels = tokenizer(examples['Answer'], max_length=512,
                       truncation=True, padding="max_length").input_ids
    labels_with_ignore_index = [[(label if label != tokenizer.pad_token_id else -100)
                                 for label in label_seq] for label_seq in labels]

    model_inputs["labels"] = labels_with_ignore_index
    model_inputs["decoder_input_ids"] = [[tokenizer.pad_token_id] + label_seq[:-1]
                                         # Создание decoder_input_ids
                                         for label_seq in labels_with_ignore_index]

    return model_inputs


tokenized_dataset = dataset.map(tokenize_function, batched=True)

def get_tokenized_dataset():
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print(tokenized_dataset)
    return tokenized_dataset
