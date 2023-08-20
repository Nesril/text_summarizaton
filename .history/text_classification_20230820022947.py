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
from transformers import pipeline, set_seed

set_seed(42)
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

print("\n\t--- Smaple Text with 1000 char\n",sample_text)
# We'll collect the generated summaries of each model in a dictionary
summaries = {}

def texte_summerization(sample_text):

    def baseline_summary_three_sent(text):
        return "\n".join(sent_tokenize(text)[:3])
    summaries['baseline'] = baseline_summary_three_sent(sample_text)

    pipe = pipeline('text-generation', model = 'gpt2-medium' )
    gpt2_query = sample_text + "\nTL;DR:\n"
    pipe_out = pipe(gpt2_query, max_length = 512, clean_up_tokenization_spaces = True)
    summaries['gpt2'] = "\n".join(sent_tokenize(pipe_out[0]["generated_text"][len(gpt2_query) :]))

    pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    pipe_out = pipe(sample_text)
    summaries["bart"] = "\n".join(sent_tokenize(pipe_out[0]["summary_text"]))

    pipe = pipeline('summarization', model="google/pegasus-cnn_dailymail"  )
    pipe_out = pipe(sample_text)
    summaries["pegasus"] = pipe_out[0]["summary_text"].replace(" .<n>", ".\n")

        
texte_summerization(sample_text)

print("\n\t GROUND TRUTH of sample text\n")
print(dataset['train'][1]['highlights'])

for model_name in summaries:
     print("\n\t",model_name.upper(),"\n")
     print(summaries[model_name])