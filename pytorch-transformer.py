from logging import config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import os
from tqdm import tqdm
import time


# Configuration dictionary - all hyperparameters in one place 
CONFIG = {
    'vocab_size': 18000,        # Vocabulary size for both source and target
    'd_model': 512,             # Model dimension (embedding size)
    'dff': 2048,                # Feed-forward network dimension
    'num_heads': 8,             # Number of attention heads
    'num_encoder_layers': 6,     # Number of encoder layers
    'num_decoder_layers': 6,     # Number of decoder layers
    'dropout_rate': 0.1,        # Dropout rate
    'max_length': 200,          # Maximum sequence length
    'batch_size': 32,           # Batch size for training
    'learning_rate': 0.0001,    # Learning rate
    'epochs': 300,               # Number of training epochs
    'apply_early_stop': True,
    'patience': 3,              # Early stopping patience
    'max_sentences': 39,    # Maximum number of sentences to read from CSV
    #'device': 'mps' if torch.backends.mps.is_available() else 'cpu'  # Use MPS on Mac M4
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
}

print(f"Using device: {CONFIG['device']}")
print(f"Configuration: {CONFIG}")
 
class Vocabulary:
    """
    Vocabulary class to handle word-to-index and index-to-word mappings
    This is essential for converting text to numbers that the model can process
    """
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_count = Counter() #  The Counter object is a specialized dictionary subclass for counting hashable object

    def build_vocab(self, sentences, max_vocab_size):
        """Build vocabulary from list of sentences"""
        # Count word frequencies
        for sentence in sentences:
            self.word_count.update(sentence.split())

        # Get most common words
        most_common = self.word_count.most_common(max_vocab_size - 4)  # -4 for special tokens

        # Add to vocabulary
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence):
        """Convert sentence to list of indices"""
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence.split()]

    def decode(self, indices):
        """Convert list of indices back to sentence"""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

    def __len__(self):
        return len(self.word2idx)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding adds position information to embeddings
    Since transformers don't have inherent position awareness like RNNs,
    we need to explicitly add position information
    """
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1) #

        # Create div_term for sine and cosine functions
        # We can implement Original paper uses 1 / (10000^(2*i/d_model))
        # This is equivalent to exp( -(2*i/d_model) * log(10000) )this directly: (a^x=e^(ln(a)))
        # torch.arange(0, d_model, 2).float()  = [0., 2., 4., ..., dmodel - 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even positions and cosine to odd positions
        pe[:, 0::2] = torch.sin(position * div_term) # select all rows, and select columns starting from index 0, going to the end, with a step of 2
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # its shape becomes [1, max_length, d_model].
         #This is done to add a "batch" dimension, even though the positional encoding is typically applied to each item in a batch identically.
         #The subsequent .transpose(0, 1) then changes the shape to [max_length, 1, d_model].
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings"""
        return x + self.pe[:x.size(0), :] # pos_encoding[:seq_len] slices the positional encodings to match the input length


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism - the core of the transformer
    It allows the model to attend to different parts of the sequence simultaneously
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention
        Attention(Q,K,V) = softmax(QK^T/√d_k)V
        """
        # K tensor has a shape of [batch_size, num_heads, sequence_length, d_k]
        # K.transpose(-2, -1) is swapping the second-to-last dimension (sequence_length) and the last dimension (d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (for padding or future positions)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        #applying softmax along the last dimension means that for each query position, each head and each sequence
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations and split into heads
        # (batch_size, seq_len, d_model) → (batch_size, seq_len, num_heads, d_k) → (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        # (batch_size, num_heads, seq_len, d_k) → (batch_size, seq_len, num_heads, d_k) → (batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        # Final linear transformation
        output = self.W_o(attention_output)
        return output

# Linear weights (W_q, W_k, W_v, W_o) are [d_model, d_model],
# while Q, K, and V before attention are [batch_size, num_heads, sequence_length, d_k],
# the final output is [batch_size, sequence_length, d_model]


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    Two linear transformations with ReLU activation in between
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, dff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer containing:
    1. Multi-head self-attention
    2. Residual connection + Layer normalization
    3. Feed-forward network
    4. Residual connection + Layer normalization
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # Multi-head attention + residual connection + layer norm
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # Residual connection

        # Feed-forward network + residual connection + layer norm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection

        return out2



class Encoder(nn.Module):
    """
    Complete Encoder consisting of:
    1. Input embedding
    2. Positional encoding
    3. Stack of encoder layers
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dff, max_length, dropout_rate):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        seq_len = x.size(1)

        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scale embeddings
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        # Pass through encoder layers
        for enc_layer in self.enc_layers:
            x = enc_layer(x, mask)

        return x
    


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer containing:
    1. Multi-head self-attention (masked)
    2. Residual connection + Layer normalization
    3. Multi-head cross-attention with encoder output
    4. Residual connection + Layer normalization
    5. Feed-forward network
    6. Residual connection + Layer normalization
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # Self-attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # Cross-attention
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # Masked self-attention + residual connection + layer norm
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        # Cross-attention + residual connection + layer norm
        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        # Feed-forward network + residual connection + layer norm
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
    
class Decoder(nn.Module):
    """
    Complete Decoder consisting of:
    1. Target embedding
    2. Positional encoding
    3. Stack of decoder layers
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dff, max_length, dropout_rate):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        seq_len = x.size(1)

        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        # Pass through decoder layers
        for dec_layer in self.dec_layers:
            x = dec_layer(x, enc_output, look_ahead_mask, padding_mask)

        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence translation
    Combines encoder and decoder with final linear layer for vocabulary prediction
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads,
                 dff, max_length, dropout_rate):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads,
                              dff, max_length, dropout_rate)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads,
                              dff, max_length, dropout_rate)
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        self.device = torch.device(CONFIG['device'])

    def create_padding_mask(self, seq):
        """Create padding mask to ignore padding tokens"""
        return (seq != 0).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        """Create look-ahead mask to prevent seeing future tokens during training"""
        mask = torch.tril(torch.ones(size, size)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt, training=True):
        # Create masks
        src_mask = self.create_padding_mask(src)

        """
        if training:
            tgt_seq_len = tgt.size(1)
            look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len).to(self.device)
            tgt_padding_mask = self.create_padding_mask(tgt).to(self.device)
            combined_mask = torch.max(look_ahead_mask, ~tgt_padding_mask)
        else:
            #combined_mask = None
            tgt_seq_len = tgt.size(1)
            look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len).to(self.device)
            combined_mask = look_ahead_mask
        """

        # Create target padding mask
        tgt_padding_mask = self.create_padding_mask(tgt).to(self.device)

        # Create look-ahead mask for target sequence
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len).to(self.device)

        # Combine look-ahead and padding masks for decoder self-attention
        combined_mask = torch.logical_and(look_ahead_mask, tgt_padding_mask)

        # Encoder
        enc_output = self.encoder(src, src_mask)

        # Decoder
        dec_output = self.decoder(tgt, enc_output, combined_mask, src_mask)

        # Final linear layer
        final_output = self.final_layer(dec_output)

        return final_output


class TranslationDataset(Dataset):
    """
    Custom Dataset class for handling translation data
    Efficiently loads and processes data in batches
    """
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]

        # Encode sentences
        src_encoded = self.src_vocab.encode(src_sentence)
        #tgt_encoded = [self.tgt_vocab.word2idx['<SOS>']] + self.tgt_vocab.encode(tgt_sentence) + [self.tgt_vocab.word2idx['<EOS>']]
        tgt_encoded_ip = [self.tgt_vocab.word2idx['<SOS>']] + self.tgt_vocab.encode(tgt_sentence)
        tgt_encoded_op = self.tgt_vocab.encode(tgt_sentence) + [self.tgt_vocab.word2idx['<EOS>']]


        # Truncate if too long
        if len(src_encoded) > self.max_length:
            src_encoded = src_encoded[:self.max_length]
        if len(tgt_encoded_ip) > self.max_length:
            tgt_encoded_ip = tgt_encoded_ip[:self.max_length]
        if len(tgt_encoded_op) > self.max_length:
            tgt_encoded_op = tgt_encoded_op[:self.max_length]


        # Pad sequences
        src_padded = src_encoded + [0] * (self.max_length - len(src_encoded))
        tgt_encoded_ip = tgt_encoded_ip + [0] * (self.max_length - len(tgt_encoded_ip))
        tgt_encoded_op = tgt_encoded_op + [0] * (self.max_length - len(tgt_encoded_op))

        return {
            'src': torch.tensor(src_padded, dtype=torch.long),
            'tgt_ip': torch.tensor(tgt_encoded_ip, dtype=torch.long),
            'tgt_op': torch.tensor(tgt_encoded_op, dtype=torch.long)
        }


def load_data_efficiently(file_name, max_sentences):
    """
    Efficiently load data from CSV file in chunks to handle large datasets
    This prevents memory issues with very large files
    """
    print(f"Loading data from {file_name}...")

    # Read data in chunks to handle large files
    chunk_size = 10000
    english_sentences = []
    bengali_sentences = []

    try:
        # Read CSV in chunks
        chunk_iter = pd.read_csv(file_name, chunksize=chunk_size)
        total_loaded = 0

        for chunk in chunk_iter:
            if total_loaded >= max_sentences:
                break

            # Filter out NaN values and empty strings
            chunk = chunk.dropna()
            chunk = chunk[chunk['en'].str.len() > 0]
            chunk = chunk[chunk['bn'].str.len() > 0]

            # Add to lists
            remaining = max_sentences - total_loaded
            chunk_to_add = min(len(chunk), remaining)

            english_sentences.extend(chunk['en'].iloc[:chunk_to_add].tolist())
            bengali_sentences.extend(chunk['bn'].iloc[:chunk_to_add].tolist())

            total_loaded += chunk_to_add
            print(f"Loaded {total_loaded} sentences...")

    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

    print(f"Successfully loaded {len(english_sentences)} sentence pairs")
    #print(f"English sentences: {english_sentences[:5]}")
    #print(f"Bengali sentences: {bengali_sentences[:5]}")
    return english_sentences, bengali_sentences

def create_data_loaders(english_sentences, bengali_sentences, src_vocab, tgt_vocab, config):
    """Create train and validation data loaders"""
    # Split data into train and validation
    split_idx = int(0.9 * len(english_sentences))

    train_src = english_sentences[:split_idx]
    train_tgt = bengali_sentences[:split_idx]
    val_src = english_sentences[split_idx:]
    val_tgt = bengali_sentences[split_idx:]

    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, config['max_length'])
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab, config['max_length'])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    Stops training when validation loss stops improving
    """
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

def train_model(model, train_loader, val_loader, config):
    """
    Train the transformer model with early stopping
    """
    device = torch.device(config['device'])
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])

    # Training history
    train_losses = []
    val_losses = []

    start_time = time.time()
    print("Starting training...")
    for epoch in range(config['epochs']):

        time_elapse = time.time() - start_time
        print(f"Time elapsed: {time_elapse // 60:.0f}m {time_elapse % 60:.0f}s")

        if time_elapse > 1 * (60 * 60):
            print("Time limit exceeded. Stopping training.")
            break


        model.train()
        epoch_train_loss = 0

        # Training loop
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            src = batch['src'].to(device)
            tgt_ip = batch['tgt_ip'].to(device)
            tgt_op = batch['tgt_op'].to(device)

            #print("\nsrc:", src)
            #print("tgt_input:", tgt_ip)
            #print("tgt_real:", tgt_op)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(src, tgt_ip, training=True)

            #print("\npredictions:", predictions)

            #print("")

            # Calculate loss
            loss = criterion(predictions.reshape(-1, predictions.size(-1)), tgt_op.reshape(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)

                tgt_ip = batch['tgt_ip'].to(device)
                tgt_op = batch['tgt_op'].to(device)

                predictions = model(src, tgt_ip, training=True)
                loss = criterion(predictions.reshape(-1, predictions.size(-1)), tgt_op.reshape(-1))
                epoch_val_loss += loss.item()

        #print("len(train_loader): ", len(train_loader))
        #print("len(val_loader):", len(val_loader))

        # Average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if config['apply_early_stop'] :
          # Early stopping check
          if early_stopping(avg_val_loss):
              print(f"Early stopping triggered at epoch {epoch+1}")
              break

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, train_losses, val_losses


class TranslationInference:
    """
    Inference class for translating sentences using trained model
    """
    def __init__(self, model, src_vocab, tgt_vocab, config):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config
        self.device = torch.device(config['device'])
        self.model.eval()

    def translate(self, sentence, max_length=None):
        """
        Translate a single English sentence to Bengali
        """
        if max_length is None:
            max_length = self.config['max_length']

        # Encode source sentence
        src_encoded = self.src_vocab.encode(sentence.lower().strip())
        if len(src_encoded) > max_length:
            src_encoded = src_encoded[:max_length]

        # Pad source
        src_padded = src_encoded + [0] * (max_length - len(src_encoded))
        src_tensor = torch.tensor([src_padded], dtype=torch.long).to(self.device)

        # Start with SOS token
        tgt_input = [self.tgt_vocab.word2idx['<SOS>']]

        with torch.no_grad():
            for _ in range(max_length):
                # Pad target input
                tgt_padded = tgt_input + [0] * (max_length - len(tgt_input))
                tgt_tensor = torch.tensor([tgt_padded], dtype=torch.long).to(self.device)

                # Forward pass
                predictions = self.model(src_tensor, tgt_tensor, training=False)

                # Get next token prediction
                next_token_logits = predictions[0, len(tgt_input)-1, :]
                next_token = torch.argmax(next_token_logits).item()

                # Add to target input
                tgt_input.append(next_token)

                # Stop if EOS token is generated
                if next_token == self.tgt_vocab.word2idx['<EOS>']:
                    break

        # Decode the result (excluding SOS and EOS tokens)
        result_tokens = tgt_input[1:-1] if tgt_input[-1] == self.tgt_vocab.word2idx['<EOS>'] else tgt_input[1:]
        translated_sentence = self.tgt_vocab.decode(result_tokens)

        return translated_sentence


def main():
    """
    Main function to orchestrate the entire training and inference process
    """
    print("=== English to Bengali Transformer Translation ===")
    print(f"Configuration: {CONFIG}")

    # Load data
    file_name = 'D:/Work/wsl/ML-DS/english_to_bangla.csv'
    english_sentences, bengali_sentences = load_data_efficiently(file_name, CONFIG['max_sentences'])

    if len(english_sentences) == 0:
        print("No data loaded. Please check the CSV file.")
        return

    # Build vocabularies
    print("Building vocabularies...")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    src_vocab.build_vocab(english_sentences, CONFIG['vocab_size'])
    tgt_vocab.build_vocab(bengali_sentences, CONFIG['vocab_size'])

    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")

    #print(f"Source vocabulary : {src_vocab.idx2word}")
    #print(f"Target vocabulary: {tgt_vocab.idx2word}")

    #print("english_sentences:", english_sentences)
    #print("bengali_sentences", bengali_sentences)
    

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        english_sentences, bengali_sentences, src_vocab, tgt_vocab, CONFIG
    )

    # Create model
    print("Creating transformer model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=CONFIG['d_model'],
        num_layers=CONFIG['num_encoder_layers'],  # Same for both encoder and decoder
        num_heads=CONFIG['num_heads'],
        dff=CONFIG['dff'],
        max_length=CONFIG['max_length'],
        dropout_rate=CONFIG['dropout_rate']
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader, CONFIG)

    # Save model and vocabularies
    print("Saving model and vocabularies...")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }, 'en_bn_transformer.pth')

    # Create inference object
    translator = TranslationInference(trained_model, src_vocab, tgt_vocab, CONFIG)

    # Test translations
    print("\n=== Testing Translations ===")
    test_sentences = [
        "a dog running"
        ,"Hello, how are you?",
        "I love you.",
        "What is your name?",
        "Good morning.",
        "Thank you very much."
    ]

    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"English: {sentence}")
        print(f"Bengali: {translation}")
        print("-" * 50)

    print("Training and inference completed successfully!")


def load_pretrained_model(model_path='en_bn_transformer.pth'):
    """
    Load a pretrained model for inference only
    """
    checkpoint = torch.load(model_path, map_location='cpu')

    config = checkpoint['config']
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']

    # Recreate model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_layers=config['num_encoder_layers'],
        num_heads=config['num_heads'],
        dff=config['dff'],
        max_length=config['max_length'],
        dropout_rate=config['dropout_rate']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create translator
    translator = TranslationInference(model, src_vocab, tgt_vocab, config)

    return translator


def print_model_architecture():
    """
    Print detailed model architecture and component counts
    """
    print("\n=== TRANSFORMER ARCHITECTURE ANALYSIS ===")

    # Create a sample model for analysis
    sample_config = CONFIG.copy()
    sample_config['vocab_size'] = 1000  # Smaller for demo

    model = Transformer(
        src_vocab_size=sample_config['vocab_size'],
        tgt_vocab_size=sample_config['vocab_size'],
        d_model=sample_config['d_model'],
        num_layers=sample_config['num_encoder_layers'],
        num_heads=sample_config['num_heads'],
        dff=sample_config['dff'],
        max_length=sample_config['max_length'],
        dropout_rate=sample_config['dropout_rate']
    )

    print("\n1. ENCODER ARCHITECTURE:")
    print(f"   - Input Embedding: {sample_config['vocab_size']} → {sample_config['d_model']}")
    print(f"   - Positional Encoding: {sample_config['d_model']} dimensions")
    print(f"   - Number of Encoder Layers: {sample_config['num_encoder_layers']}")
    print("   - Each Encoder Layer contains:")
    print(f"     * Multi-Head Attention ({sample_config['num_heads']} heads)")
    print(
        f"     * Feed-Forward Network ({sample_config['d_model']} → {sample_config['dff']} → {sample_config['d_model']})")
    print("     * 2 Layer Normalizations")
    print("     * 2 Residual Connections")

    print("\n2. DECODER ARCHITECTURE:")
    print(f"   - Target Embedding: {sample_config['vocab_size']} → {sample_config['d_model']}")
    print(f"   - Positional Encoding: {sample_config['d_model']} dimensions")
    print(f"   - Number of Decoder Layers: {sample_config['num_decoder_layers']}")
    print("   - Each Decoder Layer contains:")
    print(f"     * Masked Multi-Head Self-Attention ({sample_config['num_heads']} heads)")
    print(f"     * Multi-Head Cross-Attention ({sample_config['num_heads']} heads)")
    print(
        f"     * Feed-Forward Network ({sample_config['d_model']} → {sample_config['dff']} → {sample_config['d_model']})")
    print("     * 3 Layer Normalizations")
    print("     * 3 Residual Connections")

    print("\n3. OUTPUT LAYER:")
    print(f"   - Final Linear Layer: {sample_config['d_model']} → {sample_config['vocab_size']}")
    print("   - Softmax activation for probability distribution")

    # Count parameters for each component
    total_params = sum(p.numel() for p in model.parameters())

    # Encoder parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    final_layer_params = sum(p.numel() for p in model.final_layer.parameters())

    print("\n4. PARAMETER COUNTS:")
    print(f"   - Encoder Parameters: {encoder_params:,}")
    print(f"   - Decoder Parameters: {decoder_params:,}")
    print(f"   - Final Layer Parameters: {final_layer_params:,}")
    print(f"   - Total Parameters: {total_params:,}")

    print("\n5. ATTENTION MECHANISM DETAILS:")
    print(f"   - Number of Attention Heads: {sample_config['num_heads']}")
    print(f"   - Attention Head Dimension: {sample_config['d_model'] // sample_config['num_heads']}")
    print("   - Self-Attention: Query, Key, Value from same source")
    print("   - Cross-Attention: Query from decoder, Key & Value from encoder")
    print("   - Scaled Dot-Product: Attention(Q,K,V) = softmax(QK^T/√d_k)V")

    print("\n6. TRAINING FEATURES:")
    print(f"   - Dropout Rate: {sample_config['dropout_rate']}")
    print(f"   - Maximum Sequence Length: {sample_config['max_length']}")
    print("   - Teacher Forcing during training")
    print("   - Look-ahead mask for decoder self-attention")
    print("   - Padding mask for variable-length sequences")
    print("   - Early Stopping with patience")


def create_architecture_diagram():
    """
    Create a visual representation of the transformer architecture
    """
    print("\n=== TRANSFORMER ARCHITECTURE DIAGRAM ===")
    print("""
    INPUT (English)              TARGET (Bengali)
         |                           |
    [Embedding]                 [Embedding]
         |                           |
    [Pos. Encoding]             [Pos. Encoding]
         |                           |
    ┌─────────────┐              ┌─────────────┐
    │  ENCODER 1  │              │  DECODER 1  │
    │ ┌─────────┐ │              │ ┌─────────┐ │
    │ │Multi-Head│ │              │ │Masked   │ │
    │ │Self-Attn │ │              │ │Self-Attn│ │
    │ └─────────┘ │              │ └─────────┘ │
    │ ┌─────────┐ │              │ ┌─────────┐ │
    │ │   FFN   │ │      ┌──────▶│ │Cross-   │ │
    │ └─────────┘ │      │       │ │Attention│ │
    └─────────────┘      │       │ └─────────┘ │
         │               │       │ ┌─────────┐ │
    ┌─────────────┐      │       │ │   FFN   │ │
    │  ENCODER 2  │      │       │ └─────────┘ │
    │ ┌─────────┐ │      │       └─────────────┘
    │ │Multi-Head│ │      │              │
    │ │Self-Attn │ │      │       ┌─────────────┐
    │ └─────────┘ │      │       │  DECODER 2  │
    │ ┌─────────┐ │      │       │ ┌─────────┐ │
    │ │   FFN   │ │      │       │ │Masked   │ │
    │ └─────────┘ │      │       │ │Self-Attn│ │
    └─────────────┘      │       │ └─────────┘ │
         │               │       │ ┌─────────┐ │
         └───────────────┼──────▶│ │Cross-   │ │
                         │       │ │Attention│ │
                         │       │ └─────────┘ │
                         │       │ ┌─────────┐ │
                         │       │ │   FFN   │ │
                         │       │ └─────────┘ │
                         │       └─────────────┘
                         │              │
                         │         [Linear Layer]
                         │              │
                         │          [Softmax]
                         │              │
                         │        OUTPUT (Bengali)
                         │
    Each layer includes:
    • Residual connections (+ symbols)
    • Layer normalization
    • Dropout for regularization
    """)


def explain_code_components():
    """
    Detailed explanation of each code component for educational purposes
    """
    print("\n=== CODE EXPLANATION FOR BEGINNERS ===")

    print("\n1. VOCABULARY CLASS:")
    print("   - Converts words to numbers (tokens) that the model can understand")
    print("   - word2idx: Dictionary mapping words to indices")
    print("   - idx2word: Dictionary mapping indices back to words")
    print("   - Special tokens: <PAD> (padding), <UNK> (unknown), <SOS> (start), <EOS> (end)")

    print("\n2. POSITIONAL ENCODING:")
    print("   - Transformers process all words simultaneously (parallel)")
    print("   - Need to add position information since order matters in language")
    print("   - Uses sine and cosine functions to create unique position vectors")
    print("   - Added to word embeddings to give position awareness")

    print("\n3. MULTI-HEAD ATTENTION:")
    print("   - Core mechanism that allows model to focus on different parts of input")
    print("   - Query (Q): What am I looking for?")
    print("   - Key (K): What information is available?")
    print("   - Value (V): The actual information content")
    print("   - Multiple heads allow focusing on different aspects simultaneously")
    print("   - Attention weights show which words the model is focusing on")

    print("\n4. FEED-FORWARD NETWORK:")
    print("   - Simple neural network applied to each position independently")
    print("   - Two linear layers with ReLU activation in between")
    print("   - Processes information after attention has been computed")
    print("   - Same network applied to every position")

    print("\n5. RESIDUAL CONNECTIONS:")
    print("   - Add input directly to output: output = layer(input) + input")
    print("   - Helps with training deep networks by preventing vanishing gradients")
    print("   - Allows information to flow directly through the network")

    print("\n6. LAYER NORMALIZATION:")
    print("   - Normalizes activations to have mean=0 and std=1")
    print("   - Stabilizes training and speeds up convergence")
    print("   - Applied before each sub-layer (pre-norm architecture)")

    print("\n7. MASKS:")
    print("   - Padding mask: Ignore padded positions (value 0)")
    print("   - Look-ahead mask: Prevent decoder from seeing future tokens")
    print("   - Essential for proper training and inference")

    print("\n8. TRAINING PROCESS:")
    print("   - Teacher forcing: Use correct previous tokens during training")
    print("   - Cross-entropy loss: Measures prediction accuracy")
    print("   - Adam optimizer: Adaptive learning rate algorithm")
    print("   - Early stopping: Prevents overfitting by monitoring validation loss")

    print("\n9. INFERENCE PROCESS:")
    print("   - Autoregressive generation: Generate one token at a time")
    print("   - Start with <SOS> token, generate until <EOS> or max length")
    print("   - Each generation step uses previously generated tokens")


# Additional utility functions
def demonstrate_attention_visualization():
    """
    Demonstrate how attention weights work (conceptual)
    """
    print("\n=== ATTENTION MECHANISM DEMONSTRATION ===")
    print("\nExample: Translating 'I love cats' to Bengali")
    print("\nEncoder Self-Attention might focus on:")
    print("  'I' attends to: ['I': 0.8, 'love': 0.1, 'cats': 0.1]")
    print("  'love' attends to: ['I': 0.3, 'love': 0.4, 'cats': 0.3]")
    print("  'cats' attends to: ['I': 0.1, 'love': 0.2, 'cats': 0.7]")

    print("\nDecoder Cross-Attention might focus on:")
    print("  When generating 'আমি' (I): Focus heavily on 'I' in English")
    print("  When generating 'ভালোবাসি' (love): Focus on 'love' and 'I'")
    print("  When generating 'বিড়াল' (cats): Focus on 'cats' primarily")


def show_tensor_shapes():
    """
    Demonstrate tensor shapes throughout the model
    """
    print("\n=== TENSOR SHAPES THROUGH THE MODEL ===")
    batch_size = CONFIG['batch_size']
    seq_len = CONFIG['max_length']
    d_model = CONFIG['d_model']
    vocab_size = CONFIG['vocab_size']

    print(f"\nBatch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}, Vocabulary size: {vocab_size}")

    print(f"\n1. Input tokens: [{batch_size}, {seq_len}]")
    print(f"2. After embedding: [{batch_size}, {seq_len}, {d_model}]")
    print(f"3. After positional encoding: [{batch_size}, {seq_len}, {d_model}]")
    print(f"4. Through encoder layers: [{batch_size}, {seq_len}, {d_model}]")
    print(f"5. Through decoder layers: [{batch_size}, {seq_len}, {d_model}]")
    print(f"6. After final linear: [{batch_size}, {seq_len}, {vocab_size}]")
    print(f"7. After softmax: [{batch_size}, {seq_len}, {vocab_size}] (probabilities)")


if __name__ == "__main__":
    # Print architecture details before training
    print("\n=== TRANSFORMER MODEL OVERVIEW ===")
    print("This code implements a complete Transformer model for English to Bengali translation.")  
    print_model_architecture()
    create_architecture_diagram()
    explain_code_components()
    demonstrate_attention_visualization()
    show_tensor_shapes()

    # Run main training and inference
    main()

    print("\n=== ADDITIONAL USAGE EXAMPLES ===")
    print("\n# To load a pretrained model:")
    print("translator = load_pretrained_model('en_bn_transformer.pth')")
    print("translation = translator.translate('Hello world')")
    print("print(translation)")

    print("\n# To translate multiple sentences:")
    print("sentences = ['Good morning', 'How are you?', 'Thank you']")
    print("for sentence in sentences:")
    print("    translation = translator.translate(sentence)")
    print("    print(f'{sentence} → {translation}')")

    print("\n=== MODEL OPTIMIZATION TIPS ===")
    print("1. For better performance on Mac M4:")
    print("   - Use MPS backend (Metal Performance Shaders)")
    print("   - Reduce batch size if running out of memory")
    print("   - Use mixed precision training if needed")

    print("\n2. For large datasets:")
    print("   - Implement data streaming")
    print("   - Use gradient accumulation")
    print("   - Consider model parallelism")

    print("\n3. For better translations:")
    print("   - Increase model size (d_model, num_layers)")
    print("   - Use more training data")
    print("   - Implement beam search for inference")
    print("   - Add attention visualization")

    print("\n=== TROUBLESHOOTING ===")
    print("1. If CUDA/MPS not available: Model will use CPU (slower)")
    print("2. If out of memory: Reduce batch_size or max_length")
    print("3. If poor translations: Train longer or increase model size")
    print("4. If loss not decreasing: Check learning rate and data quality")

    print("\nTransformer implementation completed!")
    print("This code provides a complete educational example of:")
    print("- Transformer architecture implementation")
    print("- English to Bengali translation")
    print("- Efficient data loading for large datasets")
    print("- Training with early stopping")
    print("- Inference pipeline for translation")
    print("- Detailed explanations for learning")