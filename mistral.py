from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers


model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo

model.eval() # model in evaluation mode (dropout modules are deactivated)



# craft prompt
request = "Please give me soothing and calm music"
intstructions_string = f"""MusicBot, functioning as a virtual bot that recommend music composition to user. Give the user the name of the song, year, album and artist.\
It analyzes user request and ends response with its signature '–MusicBot'. \

Please respond to the following request and give the recommendation:
"""

prompt=f'''[INST] {request} [/INST]'''

prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''
prompt = prompt_template(request)

print(prompt)
# tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# generate output
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)

print(tokenizer.batch_decode(outputs)[0])
