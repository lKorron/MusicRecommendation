from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, T5Tokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig
from datasets import load_dataset, DatasetDict

dataset = load_dataset("csv", data_files="rephrased_questions.csv")

train_testvalid = dataset['train'].train_test_split(test_size=0.2)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)


dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']
})


model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"eos_token": "</s>"})

task_prefix = "answer"


def tokenize_function(example):
    tokenized_data = tokenizer(
        example["Question"], text_target=example["Answer"], truncation=True, padding='max_length', max_length=128)
    return tokenized_data


tokenized_dataset = dataset.map(tokenize_function, batched=True)

peft_config = LoraConfig(task_type="SEQ_2_SEQ_LM", r=1,
                         lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.01)

lora_model = get_peft_model(model, peft_config)

lr = 3e-4
batch_size = 1
num_epochs = 2

# training_args = TrainingArguments(
#     output_dir="/Users/grigorijnikitin/.cache/huggingface/hub/preds",
#     learning_rate=lr,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=num_epochs,
#     weight_decay=0.01,
#     evaluation_strategy="no",
#     save_strategy="no",
#     load_best_model_at_end=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    "test-trainer", per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs)


trainer = Trainer(lora_model, training_args, train_dataset=tokenized_dataset["train"],
                  eval_dataset=tokenized_dataset["valid"],
                  data_collator=data_collator, tokenizer=tokenizer)
trainer.train()
