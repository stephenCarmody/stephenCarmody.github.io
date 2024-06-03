---
layout: post
title: An Introduction to the Transformer Architecture (Part 2)
categories: [AI, NLP]
---

* Do not remove this line (it will not be displayed)
{:toc}

# The Transformer 


<br>

- In 2017, Google researchers introduced the Transformer arhitecture for the purpose of translation: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- The paper had a huge impact, not just because it led to impressive results in translation accuracy, but moreso that the transformer architecture provides a way to parallelize the self-attention mechanism, which makes it much more efficient and powerful compared to previous models like RNNs and LSTMs. 

<br>

---
<p style='text-align: center; font-style: italic;'> ""..the Transformer requires less computation to train and is a much better fit for modern machine learning hardware, speeding up training by up to an order of magnitude."" </p>
---
<br>


# How it Works

- Overview 
- Tokenization
- Encoder + Embeddings 
    - Self Attention Mechanism 
    - Multi Head Attention & parellel computation
- Positional Embeddings 
- Residuals & Layer Normalization
- Decoder
- Linear & Softmax Layer

<br>

### Overview

The Transformer follows an encoder decoder arhitecture as first layed in the seimnal paper X. Here's this image below shows what a encoder block (left) and a decoder block (right) looks like. Each encoder block is repeated N times (the original paper uses 6). You might be asking yourself what's happening in each of these encoder blocks and why do we need to stack 6 of them ? Well the answer for that is....


<img src="/images/transformer/transformer_architecture.png" alt="Transformer Architecture"/>


<br>

### Tokenization 
Now that we have a very general idea how how the transformer looks like, let's zoom out and see how input data is processed before being fed into the encoder blocks. 

Raw input text to smaller blocks, via tokenization, the process of breaking down text into smaller subword units, known as tokens.

There are many different tokenizers available each with it's own pros and cons (see this HuggingFace [article](https://huggingface.co/docs/transformers/en/tokenizer_summary)). See Andrej Karpathy's [Twitter thred](https://twitter.com/karpathy/status/1759996551378940395) for more details on the challenges faced wiht Tokenizers. 

<br>

<img src="/images/transformer/tokenizer.png" alt="Tokenizer" width="500"/>


<br>

### Word & Positional Embeddings

Word Embeddings

- One hot encode 
- Vocabulary Size
- Embedding Length

Turning tokens into fixed dimension embeddings 


<br>

### Encoder & Decoder Stacks

The residual connection is a way to pass the output of one layer directly to a later layer without any modification. The purpose of this connection is to help overcome the vanishing gradient problem that can occur during training of deep neural networks.

Dropout is applied to prevent overfitting. 

<br>

### Attention

K, Q, V


<br>

### Output Layer

Softmax & Temperature

To understand better how the softmax function works you can check out this great YouTube video, [Softmax Function Explained In Depth with 3D Visuals](https://www.youtube.com/watch?v=ytbYRIN0N4g)

<br>

## Relationship to Hardware

## Limitations & Improvements 

### Quadratic Cost of Attention

One of the big problems with transformers is the quadratic cost of attention. Basically, the amount of computation needed for the attention mechanism grows really fast as the input size increases. This can make transformers very slow and expensive to run, especially with long sequences of data. 

<br>

### Improvements to the Original Transformer

- Sparse Attention & Increased Context Window
- Quantization

# Resources 
- Original Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Blog Post: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

<br>
<br>
<br>