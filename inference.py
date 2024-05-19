def get_bot_response(request, context,  model, tokenizer):

    model.eval()

    prompt_template = lambda request, context: f"""[INST]MusicBot, functioning as a virtual bot that recommend music composition to user. \
    It ends response with its signature '–MusicBot'. \

    {context}
    Please respond to the following request and give the recommendation.  Use the data about songs from context. Do not mention that you use the context.

    {request}
    [/INST]
    """

    prompt = prompt_template(request, context)

    print(prompt)
    # tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # generate output
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)

    print(tokenizer.batch_decode(outputs)[0])
