# void_text_summarizaton
I use the word void b/z I didn't use any kind of libraries or datasets.
This code is for Executive Summarization. let me say you s.thing about Executive Summarization

# Extractive Summarization: 
an approach that selects the most important phrases and lines from the documents. It then combines all the important lines to create the summary. So, in this case, every line and word of the summary actually belongs to the original document which is summarized.

In extractive summarization, the goal is to select the most salient and relevant phrases or sentences from the source document and combine them to create a summary. Unlike abstractive summarization, where the summary may contain new words or phrases not present in the source document, extractive summarization strictly relies on extracting and rearranging existing content from the original text.

The extractive approach identifies important lines or sentences based on various criteria such as word frequency, sentence position, or similarity to the overall document. These important lines are then concatenated to form the summary. Since the extracted lines are directly taken from the original document, every word and phrase in the summary can be traced back to the source document.

Extractive summarization aims to preserve the factual information and context of the original document. It can be an effective approach for generating summaries that faithfully represent the key points of the source material. However, extractive methods may sometimes result in summaries that lack coherence or fail to capture the overall meaning of the document, especially when dealing with longer texts or documents with complex structures.

## Main.py is about Extractive Summarization:

# Abstractive summarization:
is a more advanced approach compared to extractive summarization. In abstractive summarization, the goal is to generate a concise summary that captures the key information from the original document using new phrases and terms. This approach involves understanding the meaning and context of the document and then generating a summary that may not necessarily be present in the original text.

Abstractive summarization often requires more advanced natural language processing techniques, such as language generation models like recurrent neural networks (RNNs) or transformer-based models like the ones used in state-of-the-art language models such as GPT (Generative Pre-trained Transformer) and BERT (Bidirectional Encoder Representations from Transformers).

Compared to extractive summarization, abstractive summarization has the advantage of being able to generate concise and coherent summaries that can capture the essence of the original document in a more human-like manner. However, it is also a more challenging task as it requires a deeper understanding of the text and the ability to generate new, grammatically correct and semantically meaningful sentences.

While extractive summarization relies on selecting and rearranging existing sentences or phrases from the source document, abstractive summarization goes beyond that by generating new sentences that may not exist in the original document but still convey the same meaning.

Overall, abstractive summarization is a more advanced and complex task due to its reliance on natural language generation and understanding techniques, but it offers the potential for more informative and concise summaries.

## the file text_classification.py is about Abstractive summarization

## I have tried to use different pretrained models like gpt-2, tf5, bart and pegasus.
