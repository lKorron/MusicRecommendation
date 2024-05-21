# Music recommendation system

![img.png](images/img.png)

This bot can give you recommendation base on corresponding request. It gives name of song,
album, artist and even short description of track, with notes why you may like it.

## Inference expamples
     User: Can you recommend me the combination of slipknot and metallica
     Bot: I'd recommend checking out "Corrupt TV" from the album "Lunatic" by Pitbulls In The Nursery (2006) – it's a fusion
     of industrial metal and progressive metal, and it showcases both heavy metal and alternative metal influences, much like
     the combination of Slipknot and Metallica in its aggressive and intricate sound. The song's industrial metal edge is
     reminiscent of Metallica's heavier works, while its progressive metal elements bring to mind the intricacy and complexity
     of Slipknot's progressive metal influences. –MusicBot.

     User: Can you recommend me a soothing music for night dreaming
     Bot:  I recommend the track "Triple Concerto" from the album "Pieces In A Modern Style" by William Orbit. This instrumental
     piece offers a lush, calming atmosphere that's perfect for nighttime dreaming. With its electronic and classical crossover
     elements, it creates a soothing and reverent ambiance that's both gentle and expansive. –MusicBot.</s>

## How to use it
You can start bot from google collab, using commands as follows:
```python
!git clone https://github.com/lKorron/MusicRecommendation
!pip install llama-index
!pip install llama-index-embeddings-huggingface
!pip install peft
!pip install auto-gptq
!pip install optimum
!pip install bitsandbytes
!pip uninstall torch -y
!pip install torch==2.1
!python3 main.py
```
If you want to use model locally, you need to:
1) Install python 3.10 (it doesn't work with 3.12)
2) Install all the dependincies like before, expect bitsandbytes:
```python
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.0-py3-none-win_amd64.whl
```
3) Finde config.json file in model folder (usually it lays in .cache directory) and add this string into quantization section:
```json
"disable_exllama": true
```

After installing dependencies you can use your own prompt in config.py file:
```python
# Inference setup
"request": "Can you recommend me the combination of slipknot and metallica"
```
## Training
Also you can set your own hparams in config.py file. To train new model setup corresponding flag
go True. Diactivate loding tuned model and run the main.py file. 

```python
"is_train": True,
"is_load_tuned": False,
```

If you like the result you can use tuned model with chosen checkpoint:

```python
"is_train": False,
"is_load_tuned": True,
"model_checkpoint": "MusicBot-ft/checkpoint-24",
```

## Team
- Grigorij Nikitin
- Daniilь Ibragimov
- Yurchenko Anna
- Busьko Polina