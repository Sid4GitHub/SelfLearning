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
import random
import re  # For text normalization


# Configuration dictionary - all hyperparameters in one place 
CONFIG = {
    'vocab_size': 18000,        # Maximum vocabulary size for both source and target languages
    'd_model': 512,             # Model dimension (embedding size for each token)
    'dff': 2048,                # Feed-forward network dimension (hidden size in FFN)
    'num_heads': 8,             # Number of attention heads in multi-head attention
    'num_encoder_layers': 6,    # Number of encoder layers (stacked)
    'num_decoder_layers': 6,    # Number of decoder layers (stacked)
    'dropout_rate': 0.1,        # Dropout rate for regularization
    'max_length': 200,          # Maximum sequence length for input/output
    'batch_size': 32,           # Batch size for training
    'pretrain_learning_rate': 0.0001,    # Learning rate for pre-training
    'finetune_learning_rate': 0.00005,   # Learning rate for fine-tuning (lower)
    'pretrain_epochs': 500,               # Number of pre-training epochs
    'finetune_epochs': 50,              # Number of fine-tuning epochs
    'apply_early_stop': True,           # Whether to use early stopping
    'patience': 3,                      # Early stopping patience (epochs)
    'english_file': 'D:/Work/wsl/ML-DS/EBook_of_The_Bhagavad-Gita_English.txt',       # Path to English monolingual data
    'bengali_file': 'D:/Work/wsl/ML-DS/EBook_of_The_Bhagavad-Gita_Bengali.txt',       # Path to Bengali monolingual data
    'translation_file': 'D:/Work/wsl/ML-DS/english_to_bangla.csv',                    # Path to parallel translation data
    'max_pretrain_sentences': 5000,    # Maximum sentences for pre-training (for quick test)
    'max_translation_pairs': 5000,      # Maximum translation pairs for fine-tuning (for quick test)
    'mask_probability': 0.15,            # Probability of masking tokens during pre-training
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    'max_train_minutes': 15,           # Maximum allowed training time (in minutes) for each phase
    'max_global_minutes': 30,          # Maximum allowed total wall-clock time (in minutes) for the whole script
}

print(f"Using device: {CONFIG['device']}")
print(f"Configuration: {CONFIG}")
 
def normalize(sentence):
    """Lowercase and remove punctuation for consistent tokenization."""
    return re.sub(r'[^\w\s]', '', sentence.lower().strip())

class Vocabulary:
    """
    Vocabulary class to handle word-to-index and index-to-word mappings.
    Handles special tokens and unknown words.
    Now uses normalization for consistent tokenization.
    """
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3, '<MASK>': 4}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>', 4: '<MASK>'}
        self.word_count = Counter()

    def build_vocab(self, sentences, max_vocab_size):
        """Build vocabulary from list of sentences, using normalization."""
        for sentence in sentences:
            if isinstance(sentence, str):
                self.word_count.update(normalize(sentence).split())
            else:
                for sent in sentence:
                    self.word_count.update(normalize(sent).split())
        most_common = self.word_count.most_common(max_vocab_size - 5)  # -5 for special tokens
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence):
        """Convert sentence to list of indices, using normalization."""
        if isinstance(sentence, str):
            return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in normalize(sentence).split()]
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in normalize(' '.join(sentence)).split()]

    def decode(self, indices):
        """Convert list of indices back to sentence."""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

    def __len__(self):
        return len(self.word2idx)

class PositionalEncoding(nn.Module):
    """
    Adds position information to token embeddings using sine/cosine functions.
    This allows the transformer to be aware of token order.
    """
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings"""
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention mechanism.
    Allows the model to focus on different parts of the sequence simultaneously.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        output = self.W_o(attention_output)
        return output

class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network (applied to each position independently).
    Consists of two linear layers with ReLU activation in between.
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
    A single encoder layer: multi-head self-attention + feed-forward + normalization.
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
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class Encoder(nn.Module):
    """
    The encoder stack: input embedding + positional encoding + N encoder layers.
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

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, mask)

        return x

class DecoderLayer(nn.Module):
    """
    A single decoder layer: masked self-attention + encoder-decoder attention + feed-forward + normalization.
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
    
class Decoder(nn.Module):
    """
    The decoder stack: target embedding + positional encoding + N decoder layers.
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

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, enc_output, look_ahead_mask, padding_mask)

        return x

class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence translation.
    - Encoder is used for source language (English) understanding.
    - Decoder is used for target language (Bengali) generation.
    - Includes special heads for MLM pretraining.
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
        # For MLM pretraining: separate heads for English and Bengali
        self.mlm_head_src = nn.Linear(d_model, src_vocab_size)
        self.mlm_head_tgt = nn.Linear(d_model, tgt_vocab_size)

    def create_padding_mask(self, seq):
        """Create padding mask to ignore padding tokens"""
        return (seq != 0).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        """Create look-ahead mask to prevent seeing future tokens during training"""
        mask = torch.tril(torch.ones(size, size)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt, training=True):
        # Standard translation forward pass (not used for pretraining)
        src_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt).to(self.device)
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len).to(self.device)
        combined_mask = torch.logical_and(look_ahead_mask, tgt_padding_mask)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, combined_mask, src_mask)
        final_output = self.final_layer(dec_output)
        return final_output

    def mlm_encode(self, input_ids):
        """
        For encoder MLM pretraining (English):
        - Pass input through encoder
        - Project encoder output to English vocabulary size
        """
        # Assert input shape is [batch, seq]
        assert input_ids.dim() == 2, f"Encoder MLM input_ids must be 2D [batch, seq], got {input_ids.shape}"
        src_mask = self.create_padding_mask(input_ids)
        enc_output = self.encoder(input_ids, src_mask)
        # Assert encoder output shape
        assert enc_output.shape[:2] == input_ids.shape, f"Encoder output shape {enc_output.shape} does not match input {input_ids.shape}"
        return self.mlm_head_src(enc_output)

    def decoder_mlm(self, input_ids):
        """
        For decoder MLM pretraining (Bengali):
        - Pass input through decoder (as a language model)
        - Use dummy encoder output (zeros)
        - Project decoder output to Bengali vocabulary size
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        # Dummy encoder output (zeros)
        dummy_enc = torch.zeros(batch_size, seq_len, self.decoder.d_model, device=device)
        # Assert dummy encoder output shape
        assert dummy_enc.shape == (batch_size, seq_len, self.decoder.d_model), f"Dummy encoder shape {dummy_enc.shape} does not match (batch, seq, d_model)"
        look_ahead_mask = self.create_look_ahead_mask(seq_len).to(device)
        padding_mask = self.create_padding_mask(input_ids).to(device)
        combined_mask = torch.logical_and(look_ahead_mask, padding_mask)
        # Assert mask shapes
        assert look_ahead_mask.shape[-2:] == (seq_len, seq_len), f"Look ahead mask shape {look_ahead_mask.shape} does not match (1, 1, seq, seq)"
        assert padding_mask.shape[-1] == seq_len, f"Padding mask shape {padding_mask.shape} does not match seq_len {seq_len}"
        dec_out = self.decoder(input_ids, dummy_enc, combined_mask, None)
        # Assert decoder output shape
        assert dec_out.shape[:2] == input_ids.shape, f"Decoder output shape {dec_out.shape} does not match input {input_ids.shape}"
        return self.mlm_head_tgt(dec_out)

# Dataset classes
class PretrainDataset(Dataset):
    """
    Dataset for pre-training with masked language modeling (MLM).
    - Randomly masks tokens in each sentence.
    - Returns input_ids (masked) and labels (original, with -100 for unmasked).
    """
    def __init__(self, sentences, vocab, max_length, mask_prob=0.15):
        self.sentences = sentences
        self.vocab = vocab
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sentences)

    def mask_tokens(self, tokens):
        """Apply masked language modeling to tokens"""
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # -100 is ignored in loss calculation
        
        for i, token in enumerate(tokens):
            if token in [0, 2, 3]:  # Don't mask PAD, SOS, EOS
                continue
                
            if random.random() < self.mask_prob:
                labels[i] = token  # Original token for loss calculation
                
                # 80% of time, replace with MASK token
                if random.random() < 0.8:
                    masked_tokens[i] = self.vocab.word2idx['<MASK>']
                # 10% of time, replace with random token
                elif random.random() < 0.5:
                    masked_tokens[i] = random.randint(5, len(self.vocab) - 1)
                # 10% of time, keep original token
        
        return masked_tokens, labels

    def __getitem__(self, idx):
        sentence = self.sentences[idx].strip()
        
        # Encode sentence
        tokens = self.vocab.encode(sentence)
        
        # Truncate if too long
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        
        # Add SOS and EOS tokens
        tokens = [self.vocab.word2idx['<SOS>']] + tokens + [self.vocab.word2idx['<EOS>']]
        
        # Apply masking
        masked_tokens, labels = self.mask_tokens(tokens)
        
        # Pad sequences
        padded_tokens = masked_tokens + [0] * (self.max_length - len(masked_tokens))
        padded_labels = labels + [-100] * (self.max_length - len(labels))
        
        return {
            'input_ids': torch.tensor(padded_tokens, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }

class TranslationDataset(Dataset):
    """
    Dataset for translation fine-tuning.
    - Returns source and target sequences, both input and output forms.
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

# Data loading functions
def load_monolingual_data(file_path, max_sentences):
    """
    Loads monolingual text data from a file.
    Returns a list of sentences (strings).
    """
    print(f"Loading monolingual data from {file_path}...")
    sentences = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_sentences:
                    break
                line = line.strip()
                if line:  # Skip empty lines
                    sentences.append(line)
        
        print(f"Loaded {len(sentences)} sentences from {file_path}")
        print(f"First 5 sentences(load_monolingual_data): {sentences[:5]}")
        return sentences
        
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating sample data...")
        # Create sample data if file doesn't exist
        sample_sentences = []
        if 'english' in file_path.lower():
            sample_sentences = [
                "This is a sample English sentence.",
                "Machine learning is fascinating.",
                "Natural language processing helps computers understand text.",
                "Deep learning models can learn complex patterns.",
                "Transformers have revolutionized language modeling."
            ] * (max_sentences // 5)
        else:
            sample_sentences = [
                "এটি একটি নমুনা বাংলা বাক্য।",
                "মেশিন লার্নিং আকর্ষণীয়।",
                "প্রাকৃতিক ভাষা প্রক্রিয়াকরণ কম্পিউটারকে টেক্সট বুঝতে সাহায্য করে।",
                "গভীর শিক্ষার মডেলগুলি জটিল প্যাটার্ন শিখতে পারে।",
                "ট্রান্সফরমার ভাষা মডেলিংয়ে বিপ্লব ঘটিয়েছে।"
            ] * (max_sentences // 5)
        
        return sample_sentences[:max_sentences]

def load_translation_data(file_path, max_sentences):
    """
    Loads parallel translation pairs from a CSV file.
    Returns two lists: English sentences and Bengali sentences.
    """
    print(f"Loading translation data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
        df = df.head(max_sentences)
        
        english_sentences = df['en'].tolist()
        bengali_sentences = df['bn'].tolist()
        
        print(f"Loaded {len(english_sentences)} translation pairs")
        return english_sentences, bengali_sentences
        
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating sample translation data...")
        # Create sample translation data
        sample_pairs = [
            ("Hello", "হ্যালো"),
            ("Good morning", "সুপ্রভাত"),
            ("How are you?", "আপনি কেমন আছেন?"),
            ("Thank you", "ধন্যবাদ"),
            ("I love you", "আমি তোমাকে ভালোবাসি")
        ] * (max_sentences // 5)
        
        english_sentences = [pair[0] for pair in sample_pairs[:max_sentences]]
        bengali_sentences = [pair[1] for pair in sample_pairs[:max_sentences]]
        
        return english_sentences, bengali_sentences

def create_pretrain_dataloaders(english_sentences, bengali_sentences, src_vocab, tgt_vocab, config):
    """
    Creates DataLoaders for pretraining (MLM) on English and Bengali monolingual data.
    """
    print("Creating pre-training data loaders...")
    
    # Create datasets
    english_dataset = PretrainDataset(english_sentences, src_vocab, config['max_length'], config['mask_prob'])
    bengali_dataset = PretrainDataset(bengali_sentences, tgt_vocab, config['max_length'], config['mask_prob'])
    
    # Create data loaders
    english_loader = DataLoader(english_dataset, batch_size=config['batch_size'], shuffle=True)
    bengali_loader = DataLoader(bengali_dataset, batch_size=config['batch_size'], shuffle=True)
    
    print(f"English pre-training samples: {len(english_dataset)}")
    print(f"Bengali pre-training samples: {len(bengali_dataset)}")
    
    return english_loader, bengali_loader

def create_translation_dataloaders(english_sentences, bengali_sentences, src_vocab, tgt_vocab, config):
    """
    Creates DataLoaders for translation fine-tuning on parallel data.
    """
    print("Creating translation data loaders...")
    
    # Split data
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
    
    print(f"Translation training samples: {len(train_dataset)}")
    print(f"Translation validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

class EarlyStopping:
    """
    Utility for early stopping during training to prevent overfitting.
    Stops training if validation loss does not improve for 'patience' epochs.
    """
    def __init__(self, patience=5, min_delta=0.0001):
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

# === HELPER: BATCH ACCURACY ===
def batch_accuracy(logits, labels, ignore_index):
    """Compute token-level accuracy, ignoring positions with `ignore_index`."""
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)
        mask = labels != ignore_index
        if mask.sum() == 0:
            return 0.0  # avoid div/0 when all tokens are ignored (rare)
        correct = (preds == labels) & mask
        acc = correct.sum().item() / mask.sum().item()
        return acc

# === PRETRAIN MODEL: MLM on monolingual data ===
def pretrain_model(model, english_sentences, bengali_sentences, src_vocab, tgt_vocab, config, overall_start_time=None):
    """
    Pre-train the transformer model using Masked Language Modeling (MLM):
    - Encoder is pretrained on English monolingual data (MLM)
    - Decoder is pretrained on Bengali monolingual data (MLM, as a language model)
    """
    print("=== Starting Pre-training Phase ===")
    device = torch.device(config['device'])
    model = model.to(device)

    # 1. Create separate pre-training datasets and loaders for English (encoder) and Bengali (decoder)
    print("Creating pre-training datasets...")
    english_dataset = PretrainDataset(english_sentences, src_vocab, config['max_length'], config['mask_probability'])
    bengali_dataset = PretrainDataset(bengali_sentences, tgt_vocab, config['max_length'], config['mask_probability'])
    english_loader = DataLoader(english_dataset, batch_size=config['batch_size'], shuffle=True)
    bengali_loader = DataLoader(bengali_dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"English pre-training samples: {len(english_dataset)}")
    print(f"Bengali pre-training samples: {len(bengali_dataset)}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=config['pretrain_learning_rate'])
    early_stopping = EarlyStopping(patience=config['patience'])
    train_losses = []
    val_losses = []
    pretrain_accs = []
    start_time = time.time()

    for epoch in range(config['pretrain_epochs']):
        time_elapsed = time.time() - start_time
        global_time_elapsed = time.time() - overall_start_time if overall_start_time is not None else 0
        print(f"Time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s (phase), {global_time_elapsed // 60:.0f}m {global_time_elapsed % 60:.0f}s (overall)")
        if time_elapsed > config['max_train_minutes'] * 60:
            print("Time limit exceeded. Stopping pre-training.")
            break
        if overall_start_time is not None and global_time_elapsed > config['max_global_minutes'] * 60:
            print("Global time limit exceeded. Stopping pre-training.")
            break
        model.train()
        epoch_train_loss = 0
        batch_count = 0
        # --- Encoder MLM (English) ---
        for batch in tqdm(english_loader, desc=f"Encoder MLM (English) Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model.mlm_encode(input_ids)
            # Assert logits and labels shapes before loss
            assert logits.shape[:2] == labels.shape, f"Logits shape {logits.shape} and labels shape {labels.shape} do not match for loss"
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            batch_count += 1
            # Compute batch accuracy
            acc = batch_accuracy(logits, labels, -100)
            pretrain_accs.append(acc)
        # --- Decoder MLM (Bengali) ---
        for batch in tqdm(bengali_loader, desc=f"Decoder MLM (Bengali) Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model.decoder_mlm(input_ids)
            # Assert logits and labels shapes before loss
            assert logits.shape[:2] == labels.shape, f"Logits shape {logits.shape} and labels shape {labels.shape} do not match for loss"
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            batch_count += 1
            # Compute batch accuracy
            acc = batch_accuracy(logits, labels, -100)
            pretrain_accs.append(acc)
        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"Pre-train Epoch {epoch+1}/{config['pretrain_epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        # Early stopping: stop if training loss does not improve
        if config['apply_early_stop'] and early_stopping(avg_train_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    # Save pre-trained model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'pretrain_losses': (train_losses, val_losses),
        'pretrain_accs': pretrain_accs
    }, 'pretrained_transformer.pth')
    print("Pre-training completed!")
    return model, train_losses, val_losses, pretrain_accs

# === FINETUNE MODEL: Translation on parallel data ===
def finetune_model(model, english_sentences, bengali_sentences, src_vocab, tgt_vocab, config, overall_start_time=None):
    """
    Fine-tune the pretrained transformer model on parallel translation data (English→Bengali).
    """
    print("=== Starting Fine-tuning Phase ===")
    device = torch.device(config['device'])
    model = model.to(device)
    split_idx = int(0.9 * len(english_sentences))
    train_src = english_sentences[:split_idx]
    train_tgt = bengali_sentences[:split_idx]
    val_src = english_sentences[split_idx:]
    val_tgt = bengali_sentences[split_idx:]
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, config['max_length'])
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab, config['max_length'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    print(f"Fine-tuning training samples: {len(train_dataset)}")
    print(f"Fine-tuning validation samples: {len(val_dataset)}")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['finetune_learning_rate'])
    early_stopping = EarlyStopping(patience=config['patience'])
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    start_time = time.time()
    for epoch in range(config['finetune_epochs']):
        time_elapsed = time.time() - start_time
        global_time_elapsed = time.time() - overall_start_time if overall_start_time is not None else 0
        print(f"Time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s (phase), {global_time_elapsed // 60:.0f}m {global_time_elapsed % 60:.0f}s (overall)")
        if overall_start_time is not None and global_time_elapsed > config['max_global_minutes'] * 60:
            print("Global time limit exceeded. Stopping fine-tuning.")
            break
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0  # sum of batch accuracies
        batch_cnt = 0
        for batch in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}"):
            src = batch['src'].to(device)
            tgt_ip = batch['tgt_ip'].to(device)
            tgt_op = batch['tgt_op'].to(device)
            optimizer.zero_grad()
            outputs = model(src, tgt_ip, training=True)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_op.reshape(-1))
            # --- accuracy ---
            acc = batch_accuracy(outputs, tgt_op, ignore_index=0)
            epoch_train_acc += acc
            batch_cnt += 1
            # --- backprop ---
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / batch_cnt
        avg_train_acc = epoch_train_acc / batch_cnt
        # validation loop
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt_ip = batch['tgt_ip'].to(device)
                tgt_op = batch['tgt_op'].to(device)
                outputs = model(src, tgt_ip, training=True)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_op.reshape(-1))
                acc = batch_accuracy(outputs, tgt_op, ignore_index=0)
                epoch_val_loss += loss.item()
                epoch_val_acc += acc
                val_batches += 1
        avg_val_loss = epoch_val_loss / val_batches
        avg_val_acc = epoch_val_acc / val_batches
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)
        print(f"Fine-tune Epoch {epoch+1}/{config['finetune_epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train Acc : {avg_train_acc:.4f}, Val Acc : {avg_val_acc:.4f}")
        if config['apply_early_stop'] and early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'finetune_losses': (train_losses, val_losses),
        'train_accs': train_accs,
        'val_accs': val_accs
    }, 'finetuned_transformer.pth')
    print("Fine-tuning completed!")
    return model, train_losses, val_losses, train_accs, val_accs

class TranslationInference:
    """
    Inference class for translating English sentences to Bengali using the trained model.
    Handles encoding, autoregressive decoding, and output formatting.
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
        Translate a single English sentence to Bengali.
        """
        if max_length is None:
            max_length = self.config['max_length']
        # Encode source sentence
        src_encoded = self.src_vocab.encode(sentence)
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

# === MAIN FUNCTION FOR TWO-STAGE TRAINING ===
def main():
    """
    Main function to orchestrate the entire training and inference process:
    1. Load monolingual and parallel data
    2. Build vocabularies
    3. Create and pretrain the model (MLM)
    4. Fine-tune the model on translation pairs
    5. Save the model and vocabs
    6. Run inference on a new English sentence
    """
    # === Overall start time ===
    overall_start_time = time.time()
    print("=== English-Bengali Transformer: Pre-training and Fine-tuning ===")
    print(f"Configuration: {CONFIG}")

    # 1. Load monolingual data
    english_sentences = load_monolingual_data(CONFIG['english_file'], CONFIG['max_pretrain_sentences'])
    bengali_sentences = load_monolingual_data(CONFIG['bengali_file'], CONFIG['max_pretrain_sentences'])
    if not english_sentences or not bengali_sentences:
        print("Error: Could not load monolingual data. Please check 'english.txt' and 'bengali.txt'.")
        return

    # 2. Build vocabularies
    print("Building vocabularies...")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build_vocab(english_sentences, CONFIG['vocab_size'])
    tgt_vocab.build_vocab(bengali_sentences, CONFIG['vocab_size'])
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")

    # 3. Create model
    print("Creating transformer model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=CONFIG['d_model'],
        num_layers=CONFIG['num_encoder_layers'],
        num_heads=CONFIG['num_heads'],
        dff=CONFIG['dff'],
        max_length=CONFIG['max_length'],
        dropout_rate=CONFIG['dropout_rate']
    )

    # 4. Pre-train the model
    model, pretrain_train_losses, pretrain_val_losses, pretrain_accs = pretrain_model(
        model, english_sentences, bengali_sentences, src_vocab, tgt_vocab, CONFIG, overall_start_time=overall_start_time
    )

    # 5. Load translation data
    english_pairs, bengali_pairs = load_translation_data(CONFIG['translation_file'], CONFIG['max_translation_pairs'])
    if not english_pairs or not bengali_pairs:
        print("Error: Could not load translation data. Please check 'english_to_bangla.csv'.")
        return

    # 6. Fine-tune the model
    model, finetune_train_losses, finetune_val_losses, train_accs, val_accs = finetune_model(
        model, english_pairs, bengali_pairs, src_vocab, tgt_vocab, CONFIG, overall_start_time=overall_start_time
    )

    # 7. Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'pretrain_losses': (pretrain_train_losses, pretrain_val_losses),
        'finetune_losses': (finetune_train_losses, finetune_val_losses),
        'pretrain_accs': pretrain_accs,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, 'en_bn_transformer_pretrained_finetuned.pth')
    print("Model saved as 'en_bn_transformer_pretrained_finetuned.pth'")

    # === Plot metrics ===
    # 1. Pretraining loss
    plt.figure(figsize=(6,4))
    plt.plot(pretrain_train_losses, label='Pretrain Loss')
    plt.title('Pretraining Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pretrain_loss.png')

    # 2. Pretraining accuracy
    plt.figure(figsize=(6,4))
    plt.plot(pretrain_accs, label='Pretrain Accuracy')
    plt.title('Pretraining Accuracy vs Batch')
    plt.xlabel('Batch (across epochs)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pretrain_accuracy.png')

    # 3. Fine-tuning loss
    plt.figure(figsize=(6,4))
    plt.plot(finetune_train_losses, label='Train Loss')
    plt.plot(finetune_val_losses, label='Val Loss')
    plt.title('Fine-tuning Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('finetune_loss.png')

    # 4. Fine-tuning accuracy
    plt.figure(figsize=(6,4))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Fine-tuning Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('finetune_accuracy.png')

    # Show all plots
    #plt.show()

    # Show all plots non-blocking so script can continue/exit
    plt.show(block=False)
    plt.pause(15)  # Pause to allow plots to render
    plt.close('all')  # Close all figures to free resources

    # === Translation Inference ===
    print("\n=== Translation Inference ===")
    # Load model and vocabs (simulate fresh load)
    checkpoint = torch.load('en_bn_transformer_pretrained_finetuned.pth',  map_location=CONFIG['device'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    # Create inference object
    translator = TranslationInference(model, src_vocab, tgt_vocab, CONFIG)
    
    # New English sentence to translate
    # New English sentence to translate
    # Test translations
    print("\n=== Testing Translations ===")
    test_sentences = [
        "a child in a pink dress is climbing up a set of stairs in an entry way ."
        ,"a girl going into a wooden building ."
        ,"a dog is running in the snow"
        ,"a dog running"
        ,"Hello, how are you?"
        ,"a man in an orange hat starring at something ."
        ,"I love you."
        ,"a little girl climbing into a wooden playhouse ."
        ,"What is your name?"
        ,"two dogs of different breeds looking at each other on the road ."
        ,"Good morning."
        ,"Thank you very much."
        ,"Hello, how are you?",
        "I love you.",
        "What is your name?",
        "Good morning.",
        "Thank you very much.",
         "The weather is nice today."
    ]

    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"English: {sentence}")
        print(f"Bengali: {translation}")
        print("-" * 50)

    # === Print overall elapsed time ===
    overall_time_elapsed = time.time() - overall_start_time
    print(f"\nTotal elapsed time: {overall_time_elapsed // 60:.0f}m {overall_time_elapsed % 60:.0f}s")

if __name__ == "__main__":
    main()
