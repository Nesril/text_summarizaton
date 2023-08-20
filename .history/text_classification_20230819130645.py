from transformers import pipeline, set_seed

import matplotlib.pyplot as plt

import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize

#nltk.download("punkt")

# Load the dataset
#dataset = load_dataset("cnn_dailymail", version="3.0.0")

# Save the dataset
#dataset.save_to_disk("./cnn_dailymail")
#print("Dataset Saved !!!")


# Re-load the dataset
dataset = load_dataset("cnn_dailymail")
print("Dataset loaded !!!")
dataset = dataset.map(lambda example: {'text': example['text'], 'summary': example['highlights']})
dataset = dataset.to_arrow_dataset()
print(f"Features in cnn_dailymail : {dataset['train'].column_names}")