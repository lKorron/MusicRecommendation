config = {
    # Training setup
    "is_train": True,
    "lr": 3e-4,
    "batch_size": 4,
    "num_epochs": 10,
    "logging_steps": 50,
    # Inference setup
    "request": "Can you recommend me a rock music",
    "is_load_tuned": False,
    "model_checkpoint": "MusicBot-ft/checkpoint-20",
    # LoRa settings
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj"],
    # RAG settings
    "top_k": 3,
    "chunk_size": 100,
    "chunk_overlap": 25,
    "similarity_cutoff": 0.5,
}
