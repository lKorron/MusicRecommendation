import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict
from config import config
from inference import get_bot_response
from rag import get_context

dataset = load_dataset("csv", data_files="tuning_data/tuning_dataset.csv")

train_test = dataset['train'].train_test_split(test_size=0.2)


dataset = DatasetDict({
    'train': train_test['train'],
    'valid': train_test['test']
})


model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

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


if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"eos_token": "</s>"})

tokenizer.pad_token = tokenizer.eos_token

data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)


model.train()
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=config["r"],
    lora_alpha=config["lora_alpha"],
    target_modules=config["target_modules"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


lr = config["lr"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]

training_args = transformers.TrainingArguments(
    output_dir="MusicBot-ft",
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
    eval_dataset=tokenized_dataset["valid"],
    args=training_args,
    data_collator=data_collator
)

model.config.use_cache = False

if config["is_train"]:
    trainer.train()
    trainer.save_model("tuned_model")


if config["is_load_tuned"]:
    model = AutoModelForCausalLM.from_pretrained(config["model_checkpoint"],
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")


request = config["request"]
context = get_context(request, config["top_k"])

response = get_bot_response(request, context, model, tokenizer)

print(response)



