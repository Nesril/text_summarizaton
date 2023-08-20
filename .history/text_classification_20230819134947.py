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

print(summaries['baseline'])