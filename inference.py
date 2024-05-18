from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers


def get_bot_response(request, model, tokenizer):


    model.eval() # model in evaluation mode (dropout modules are deactivated)



    # craft prompt
    intstructions_string = f"""MusicBot, functioning as a virtual bot that recommend music composition to user. Gives the user the name of the song.\
    It ends response with its signature '–MusicBot'. \
    
    Please respond to the following request and give the recommendation:
    """

    prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''
    prompt = prompt_template(request)

    print(prompt)
    # tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # generate output
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)

    print(tokenizer.batch_decode(outputs)[0])
