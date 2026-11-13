# Smart Customer Query Understanding Using Embeddings

## Project Overview

This project explores how different embedding techniques represent and compare customer queries using numerical vectors.

We created a synthetic dataset of 300 customer queries and implemented Word2Vec, Sentence Transformers, BERT contextual embeddings, and sinusoidal positional encoding.

The objective is to understand how models represent individual words, complete sentences, contextual meanings, and token positions.

## Problem Statement

Customers often ask similar questions using different words. Additionally, the same word can have different meanings depending on its context.

This project investigates how embedding techniques capture semantic relationships and how positional encoding incorporates word-order information.

## Technologies Used

- Python
- Google Colab
- Pandas
- PyTorch
- Gensim (Word2Vec)
- Hugging Face Transformers
- Sentence Transformers
- Matplotlib

## Project Workflow

1. Created a synthetic dataset of 300 customer queries.
2. Performed subword tokenization using BERT's WordPiece tokenizer.
3. Trained a Word2Vec model to generate word embeddings.
4. Generated sentence embeddings using `all-MiniLM-L6-v2`.
5. Calculated cosine similarity to identify related customer queries.
6. Extracted contextual token embeddings using pretrained BERT.
7. Implemented sinusoidal positional encoding using PyTorch.

## Key Results

| Technique | Results |
|---|---|
| Word2Vec | 439 vocabulary words, 50-dimensional embeddings |
| Sentence Transformers | 300 sentence embeddings, 384 dimensions each |
| BERT | 768-dimensional contextual token embeddings |
| Positional Encoding | Demonstrated using 6 positions and 8 dimensions |

### Sentence Similarity

Input query: "How can I open a bank account?"

Most similar query: "I want to create a savings account."

Cosine similarity: **0.6909**

### Contextual Embeddings

| Comparison | Cosine Similarity |
|---|---:|
| Bank (financial vs. river) | 0.4509 |
| Charge (battery vs. fee) | 0.5207 |
| Interest (financial vs. personal) | 0.4348 |
| Bank (two similar financial contexts) | 0.9911 |

### Positional Encoding

We compared two sentences:

- "The customer paid the merchant."
- "The merchant paid the customer."

The original token embeddings were identical for the same words. After adding positional encodings, the representations differed because the token positions changed.

## Key Learnings

- Word2Vec generates fixed embeddings for individual words.
- Sentence Transformers represent complete sentences and support semantic similarity search.
- BERT generates context-dependent token representations.
- Positional encoding introduces word-order information into token representations.
- High cosine similarity does not necessarily mean identical meaning.

## Limitations

- The dataset is synthetic and relatively small.
- Word2Vec similarity results may be unreliable due to limited training data.
- Sentence similarity does not guarantee identical customer intent.
- The positional encoding experiment used randomly initialized token embeddings.
- This project demonstrates text representations and similarity matching, not a complete customer support chatbot.

## How to Run

1. Open the `.ipynb` notebook in Google Colab.
2. Install the dependencies using the installation cells provided.
3. Run the notebook cells sequentially.
4. Examine the generated embeddings, similarity scores, and positional encoding outputs.

## Conclusion

This project provides a practical understanding of word embeddings, sentence embeddings, contextual embeddings, and positional encoding.

It demonstrates how these techniques help represent customer queries numerically and form the foundation for semantic search, query matching, and Transformer-based language applications.
