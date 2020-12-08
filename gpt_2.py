# -*- coding: utf-8 -*-
"""GPT-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pk8NBEieWeEqn8d7zsGJDYwzazShkEBm
"""

!pip install simpletransformers

import pandas as pd
import numpy as np
import logging
import torch
from simpletransformers.language_modeling import LanguageModelingModel
from simpletransformers.language_generation import LanguageGenerationModel

url = 'https://github.com/lmu-mandy/project-rgt/blob/main/preprocessed_data.csv.zip?raw=true'
df = pd.read_csv(url, compression='zip', header=0, sep=',', quotechar='"', index_col=0)
print(df.shape)
df.head()

if torch.cuda.is_available():    
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

transcripts = df['transcript'].tolist()

index = int(len(transcripts) * 0.8)

# 80% training
with open("train.txt", "w") as f:
    for transcript in transcripts[:-index]:
        f.writelines(transcript + "\n")

# 20% testing
with open("test.txt", "w") as f:
    for transcript in transcripts[-index:]:
        f.writelines(transcript + "\n")

# LANGUAGE GENERATION USING A PRE-TRAINED GPT-2 MODEL
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model = LanguageGenerationModel("gpt2", "gpt2", args={"max_length": 300})

prompts = [
    "Machine learning is",
    "Hello everyone, today I'll be discussing"
]

for prompt in prompts:
    generated = model.generate(prompt, verbose=False)
    generated = '.'.join(generated[0].split('.')[:-1]) + '.'
    print('Prompt:', prompt)
    print('')
    print('Generated text:', generated)
    print('')

# EVALUATE A NON FINE-TUNED GPT-2 MODEL ON TED TALKS DATA
baseline_model = LanguageModelingModel('gpt2', 'gpt2', args={"mlm": False})
baseline_model.eval_model('test.txt')

# LANGUAGE GENERATION USING A FINE-TUNED PRE-TRAINED GPT-2 MODEL ON THE TED TALKS DATASET
train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "train_batch_size": 20,
    "num_train_epochs": 8,
    "mlm": False, 
    "learning_rate": 4e-7,
    "adam_epsilon": 1e-9
}

model = LanguageModelingModel('gpt2', 'gpt2', args=train_args)
model.train_model("train.txt", eval_file="test.txt")
model.eval_model("test.txt")

"""- Smaller batch size leads to lower loss and perplexity and results were better with num epochs >5.

lr: 4e-5  
adam epsilon: 1e-8  
bs: 20  
epochs: 8  

loss: 3.2385
perplexity: 25.4946


lr: 4e-9  
adam epsilon: 1e-10  
bs: 20
epochs: 12

loss: 3.3624
perplexity: 28.8572


lr: 4e-7  
adam epsilon: 1e-9  
bs: 20  

loss: 3.3779
perplexity: 29.3087

"""

text_generator = LanguageGenerationModel("gpt2", "./outputs", args={"max_length": 300}) # use trained weights to generate text

prompts = [
    "Machine learning is", 
    "Hello everyone, today I'll be discussing"
]

for prompt in prompts:
    generated = text_generator.generate(prompt, verbose=False)
    generated = '.'.join(generated[0].split('.')[:-1]) + '.'
    print('Prompt:', prompt)
    print('')
    print('Generated text:', generated)
    print('')

