import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict
from config import config
from inference import get_bot_response

dataset = load_dataset("csv", data_files="music_bot_dataset2.csv")

train_test_valid = dataset['train'].train_test_split(test_size=0.2)
test_valid = train_test_valid['test'].train_test_split(test_size=0.5)


dataset = DatasetDict({
    'train': train_test_valid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']
})


model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# prepare data
def tokenize_function(examples):
    text = examples["sample"]


    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# pad token should be null?

if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"eos_token": "</s>"})

tokenizer.pad_token = tokenizer.eos_token

data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

#prepare model

model.train()
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


lr = 2e-4
batch_size = 32
num_epochs = 5

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir="MusicBot-ft-2",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
)



trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    data_collator=data_collator
)

model.config.use_cache = False

# trainer.train()
#
# trainer.save_model("tuned_model")

# model = AutoModelForCausalLM.from_pretrained("MusicBot-ft-2/checkpoint-125",
#                                              device_map="auto",
#                                              trust_remote_code=False,
#                                              revision="main")


get_bot_response("Can you recommend me a rock music", model, tokenizer)



