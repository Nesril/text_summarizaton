from transformers import pipeline, set_seed

import matplotlib.pyplot as plt

import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize
import spacy
#nltk.download("punkt")
from datasets import load_dataset

nlp=spacy.load("en_core_web_lg")

dataset = load_dataset("cnn_dailymail", version="3.0.0")

print(f"Features in cnn_dailymail : {dataset['train'].column_names}")

sample = dataset["train"][1]
#print(f"""Article (excerpt of 500 characters, total length: {len(sample["article"])}):""")
#print(sample["article"][:500])
#print(f'\nSummary (length: {len(sample["highlights"])}):')
#print(sample["highlights"])

#Text Summarization Pipelines
sample_text = dataset["train"][1]["article"][:1000]

#print(sample_text)
# We'll collect the generated summaries of each model in a dictionary
summaries = {}
def baseline_summary_three_sent(text):
    return "\n".join(sent_tokenize(text)[:3])
summaries['baseline'] = baseline_summary_three_sent(sample_text)

'''
GPT-2
We can use GPT-2 it to generate summaries by simply appending “TL;DR” at the end of the input text.

The expression “TL;DR” (too long; didn’t read) is often used on platforms like Reddit to indicate a short version of a long post. 
We will start our summarization experiment by re-creating the procedure of the original paper with the pipeline() function from Transformers

We create a text generation pipeline and load the GPT-2 model:
'''
from transformers import pipeline, set_seed

set_seed(42)

pipe = pipeline('text-generation', model = 'gpt2-medium' )

gpt2_query = sample_text + "\nTL;DR:\n"

pipe_out = pipe(gpt2_query, max_length = 512, clean_up_tokenization_spaces = True)
print(pipe_out)
print("\n\t----------------------\n")
print(pipe_out[0]["generated_text"][len(gpt2_query) :])
summaries['gpt2'] = "\n".join(sent_tokenize(pipe_out[0]["generated_text"][len(gpt2_query) :]))
print("\n\t----------the array------------\n")
print("\n".join(sent_tokenize(pipe_out[0]["generated_text"][len(gpt2_query) :])))