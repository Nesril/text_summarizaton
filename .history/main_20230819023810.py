import spacy
from collections import Counter
from heapq import nlargest

nlp = spacy.load("en_core_web_lg")

def preprocess_text(text):
    doc = nlp(text)
    processed_text = []
    for token in doc:
        if not token.is_punct and not token.is_stop and not token.is_space:
            processed_text.append(token.lemma_.lower())
    return processed_text

def generate_summary(text, num_sentences=3):
    # Preprocess the text
    word_counts = Counter(preprocess_text(text))

    # Calculate word frequencies
    max_freq = max(word_counts.values())
    word_freq = {word: freq / max_freq for word, freq in word_counts.items()}
    # Tokenize sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Calculate sentence scores based on word frequencies
    sentence_scores = {sent: sum(word_freq.get(word, 0) for word in preprocess_text(sent))
                       for sent in sentences}

    # Select the top sentences for the summary
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    return ' '.join(summary_sentences)

# Example usage
text='''
Transformer-based pretrained language models (Vaswani et al., 2017; Devlin et al., 2019; Lewis et al., 2020; Raffel et al., 2020b; Brown et al., 2020) have been widely successful across all areas of natural language understanding. However, applying them over long texts (such as stories, scripts, or scientific articles) is prohibitive due to their quadratic complexity in the input length. To bridge this gap, recent work has developed more efficient transformer variants (Kitaev et al., 2020; Beltagy et al., 2020; Zaheer et al., 2020a; Guo et al., 2022) and applied them over long-range language understanding tasks (Mehta et al., 2022; Shaham et al., 2022).

However, most efficient transformers use specialized architectures with custom implementations that are not guaranteed to scale as well as vanilla transformers (Tay et al., 2022a). Moreover, they require an expensive pretraining step and do not exploit off-the-shelf pretrained LMs that were trained for short texts. To date, their performance on long texts has not matched the success of their short-range counterparts.

In this work, we present SLED: SLiding-Encoder and Decoder, a simple yet powerful method for applying off-the-shelf pretrained encoder-decoder models on long text problems, with a linear time and space dependency. Specifically (see Figure 2), we partition long documents into overlapping chunks of tokens of constant length and encode each chunk independently with an already-pretrained encoder. Then, a pretrained decoder attends to all contextualized input representations to generate the output. Our main assumption is that input tokens can be contextualized through their local surrounding (using a short-text LM), and any global cross-chunk reasoning can be handled by the decoder, similar to fusion-in-decoder (FiD) (Izacard and Grave, 2021). Our approach can be readily applied to any pretrained encoder-decoder LM such as T5 (Raffel et al., 2020b) and BART (Lewis et al., 2020) (but is not applicable to decoder-only [Brown et al., 2020] or encoder-only models [Liu et al., 2019; Conneau et al., 2020]).

We evaluate SLED on a wide range of language understanding tasks. To substantiate SLED’s adequacy for text processing, we perform controlled experiments over modified versions of SQuAD 1.1 (Rajpurkar et al., 2016) and HotpotQA (Yang et al., 2018) to show that SLED can (a) find relevant information that is embedded within a long text sequence and (b) fuse information from chunks that were encoded separately.

Our main evaluation is over SCROLLS, a recently-released benchmark that includes 7 long-range tasks across Question Answering (QA), Summarization, and Natural Language Inference (NLI). We show (Figure 1) that taking a pre-trained encoder-decoder model, such as BART (Lewis et al., 2020) or T5 (Raffel et al., 2020b), and embedding it into SLED’s framework results in dramatic improvement in performance (6 points on average across models). Moreover, BARTlarge-SLED’s performance is comparable to LongT5base (Guo et al., 2022), a model that was specifically pretrained to handle long-range dependencies, and surpasses UL2 (Tay et al., 2022b), which contains 50x more parameters. Importantly, SLED-based models can use any future pretrained LM out-of-the-box without requiring additional pretraining to further improve performance.
'''
summary = generate_summary(text)
print(summary)