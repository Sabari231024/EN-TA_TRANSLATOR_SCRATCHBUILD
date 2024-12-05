### Objective
This Project Mainly Focuses on Language - Language Translation based on the Transformer Architecture from Attention is All you Need . I have focused on bilingual language translation.

### Model
The Model is completely based on replication of "Attention is all you need Paper" - it contains the necessary components like ::
# Transformer Implementation README

## Overview
This code implements a **Transformer model** based on the architecture introduced in *"Attention Is All You Need"* by Vaswani et al. The Transformer model is widely used for sequence-to-sequence tasks like machine translation, text summarization, and more. This implementation includes all the critical components of the Transformer model, including multi-head attention, positional encoding, and residual connections.

### 1. **InputEmbedding**
- **Purpose**: Converts input tokens into dense vectors of dimension `d_model`.
- **Features**:
  - Includes a scaling factor of `sqrt(d_model)` to maintain numerical stability during training.
- **Key Method**: `nn.Embedding` creates embeddings for vocabulary tokens.

### 2. **PositionalEmbedding**
- **Purpose**: Adds positional information to the embeddings since Transformers lack inherent sequence order understanding.
- **Features**:
  - Uses sine functions for even indices and cosine functions for odd indices to generate a unique positional encoding.
  - Positional encodings are added to input embeddings.

### 3. **LayerNorm**
- **Purpose**: Normalizes the input tensor for better convergence and numerical stability.
- **Features**:
  - Includes learnable parameters `alpha` (scaling) and `bias` (shifting).

### 4. **FeedForward**
- **Purpose**: A two-layer fully connected network for processing each position's representation independently.
- **Features**:
  - Applies a ReLU activation between layers.
  - Includes dropout for regularization.

### 5. **MultiHeadAttention**
- **Purpose**: Implements scaled dot-product attention across multiple heads to learn context across sequences.
- **Features**:
  - Splits `d_model` into `h` heads, each with dimensionality `d_k = d_model / h`.
  - Computes attention scores and weighted sums for query, key, and value matrices.
  - Supports masking to prevent attending to padding tokens.
  - Implements dropout to regularize attention weights.

### 6. **Residual Connection (`ResCon`)**
- **Purpose**: Implements residual (skip) connections to avoid vanishing gradients and facilitate learning.
- **Features**:
  - Normalizes input before passing it to the sub-layer.

### 7. **EncoderBlock**
- **Purpose**: Combines self-attention, feedforward, and residual connections into a single block.
- **Features**:
  - Processes input sequentially through self-attention and feedforward layers.
  - 
### 8. **DecoderBlock**
- **Purpose**: Combines self-attention, cross-attention (to encoder outputs), and feedforward layers.
- **Features**:
  - Implements masked self-attention to prevent the decoder from attending to future positions in the sequence.
  - Cross-attention enables interaction between encoder outputs and decoder inputs.

### 9. **Encoder and Decoder**
- **Encoder**:
  - A stack of `N` `EncoderBlock`s.
  - Encodes the input sequence into a fixed-length representation.
- **Decoder**:
  - A stack of `N` `DecoderBlock`s.
  - Converts encoded representation into the target sequence.

### 10. **ProjectionLayer**
- **Purpose**: Maps the decoder's output embeddings to the target vocabulary.
- **Features**:
  - Applies a linear transformation followed by log softmax for numerical stability.

### 11. **Transformer Class**
- **Purpose**: Combines the encoder, decoder, embeddings, and projection layer into a full Transformer model.
- **Features**:
  - Encodes source sequences using the encoder.
  - Decodes target sequences using the decoder.
  - Projects the final output to predict target tokens.

### 12. **Build_transformer Function**
- **Purpose**: A utility function to assemble the Transformer with configurable hyperparameters.
- **Parameters**:
  - `src_vocab_size` and `tgt_vocab_size`: Size of source and target vocabularies.
  - `src_seq_len` and `tgt_seq_len`: Maximum sequence lengths for source and target sequences.
  - `d_model`: Dimensionality of embeddings and model representations (default: 512).
  - `N`: Number of encoder and decoder layers (default: 6).
  - `h`: Number of attention heads (default: 8).
  - `dropout`: Dropout probability (default: 0.1).
  - `d_ff`: Size of the feedforward layer (default: 2048).

### OTHER COMPONENTS ::
- I have also added other components for data preprocessing inference and training loop in multiple python files. also provided the vocab for my dataset for english and tamil.

### Dataset ::
- The dataset is taken from Hugging face :: Hemanth-thunder/en_ta 
