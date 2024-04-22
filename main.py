from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, TrainingArguments, Trainer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from datasets import load_dataset

dataset = load_dataset("csv", data_files="rephrased_questions.csv")["train"]
model = AutoModelForSeq2SeqLM.from_pretrained("EleutherAI/pile-t5-base")
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


peft_config = LoraConfig(task_type="SEQ_2_SEQ_LM", r=8,
                         lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.01)

lora_model = get_peft_model(model, peft_config)

lr = 3e-4
batch_size = 3
num_epochs = 2

training_args = TrainingArguments(
    output_dir="/Users/grigorijnikitin/.cache/huggingface/hub/preds",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="no",
    save_strategy="no",
    load_best_model_at_end=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(model=lora_model, args=training_args,
                  train_dataset=tokenized_dataset, tokenizer=tokenizer, data_collator=data_collator)
trainer.train()
