import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import urllib.request

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
# --- 1. Configuration for MiniGPT ---
CONFIG = {
    "block_size": 256,    # Maximum context length for predictions
    "batch_size": 64,
    "epochs": 5,          # More epochs might be needed for good generation
    "eval_interval": 250, # How often to evaluate
    "max_iters": 5000,    # Total training iterations
    "lr": 3e-3,
    "d_model": 384,       # Embedding dimension
    "n_heads": 6,         # Number of attention heads
    "n_layers": 2,        # Number of Transformer blocks
    "dropout": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 2. Data Loading and Preprocessing ---
print("Loading text from mtsamples.csv...")
try:
    # Make sure you have uploaded mtsamples.csv to your environment
    df = pd.read_csv('mtsamples.csv')
    # Combine all transcriptions into one long string of text
    # We drop any missing values and ensure everything is a string
    text = " ".join(df['transcription'].dropna().astype(str).tolist())
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'mtsamples.csv' not found. Please upload the file from Kaggle.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()


# Load your text data (same as before)
df = pd.read_csv('mtsamples.csv')
text_corpus = df['transcription'].dropna().astype(str).tolist()

# --- 3. Character-Level Tokenizer ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

# --- 4. Dataset and Dataloaders ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - CONFIG['block_size'], (CONFIG['batch_size'],))
    x = torch.stack([data[i:i+CONFIG['block_size']] for i in ix])
    y = torch.stack([data[i+1:i+CONFIG['block_size']+1] for i in ix])
    return x.to(CONFIG['device']), y.to(CONFIG['device'])

# --- 5. Model Implementation (Reusing and Adapting) ---

# ✨ MultiHeadAttention now uses a causal mask for GPT ✨
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Causal mask to ensure attention is only applied to the left in the input sequence
        self.register_buffer('mask', torch.tril(torch.ones(CONFIG['block_size'], CONFIG['block_size']))
                                     .view(1, 1, CONFIG['block_size'], CONFIG['block_size']))

    def forward(self, x):
        B, T, C = x.shape # Batch size, sequence length, embedding dimensionality (d_model)
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.fc_out(y)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, CONFIG['d_model'])
        self.position_embedding_table = nn.Embedding(CONFIG['block_size'], CONFIG['d_model'])
        self.blocks = nn.Sequential(*[TransformerDecoderBlock(CONFIG['d_model'], CONFIG['n_heads'], CONFIG['dropout']) for _ in range(CONFIG['n_layers'])])
        self.ln_f = nn.LayerNorm(CONFIG['d_model'])
        self.lm_head = nn.Linear(CONFIG['d_model'], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=CONFIG['device']))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates new tokens based on a starting context.
        :param idx: (B, T) tensor of indices in the current context.
        :param max_new_tokens: Number of new tokens to generate.
        :param temperature: Softmax temperature. Higher -> more random.
        :param top_k: If set, only sample from the top k most likely tokens.
        """
        for _ in range(max_new_tokens):
            # 1. CROP CONTEXT: Ensure the context is not longer than block_size.
            idx_cond = idx[:, -CONFIG['block_size']:]
            
            # 2. GET LOGITS: Get the model's predictions for the next token.
            logits, _ = self(idx_cond)
            
            # 3. FOCUS ON LAST TOKEN: We only care about the prediction for the very last token.
            logits_last_step = logits[:, -1, :] # Becomes (B, C)
            
            # 4. APPLY TEMPERATURE: Scale the logits to control randomness.
            logits_last_step = logits_last_step / temperature
            
            # 5. (OPTIONAL) APPLY TOP-K: Zero out all logits not in the top k.
            if top_k is not None:
                # Get the top k values and their indices
                topk_vals, _ = torch.topk(logits_last_step, k=top_k)
                # The k-th value is the minimum value we want to keep
                kth_val = topk_vals[:, -1]
                # Set all logits less than the k-th value to negative infinity
                logits_last_step[logits_last_step < kth_val.unsqueeze(-1)] = float('-inf')

            # 6. APPLY SOFTMAX: Convert logits to probabilities.
            probs = F.softmax(logits_last_step, dim=-1)
            
            # 7. SAMPLE: Sample the next token from the probability distribution.
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 8. APPEND: Add the sampled token to our running sequence.
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 6. Training Loop ---
model = MiniGPT().to(CONFIG['device'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])

@torch.no_grad()
def estimate_loss():
    """ Helper function to estimate loss and perplexity. """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(CONFIG['eval_interval'])
        for k in range(CONFIG['eval_interval']):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("\nStarting MiniGPT training...")
for iter in range(CONFIG['max_iters']):
    if iter % CONFIG['eval_interval'] == 0 or iter == CONFIG['max_iters'] - 1:
      losses = estimate_loss()
      val_loss = losses['val']
      val_perplexity = torch.exp(val_loss) # ✨ Calculate perplexity
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}, val perplexity {val_perplexity:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- 7. Text Generation ---
print("\n--- Generating new medical text with a specific prompt ---")

# 1. Define your desired starting text
prompt = "The patient with a history of "

# 2. Encode the prompt into token IDs
start_context_ids = encode(prompt)

# 3. Convert to a PyTorch tensor and prepare it for the model
#    - dtype=torch.long is required for embedding layers
#    - .unsqueeze(0) adds the batch dimension (shape becomes [1, sequence_length])
#    - .to(CONFIG['device']) moves the tensor to the GPU if available
start_context_tensor = torch.tensor(start_context_ids, dtype=torch.long, device=CONFIG['device']).unsqueeze(0)

# Generate text using your custom prompt
print(f"Starting prompt: {prompt}")
generated_indices = model.generate(
    idx=start_context_tensor,
    max_new_tokens=300, # Generate 300 new characters
    temperature=0.8,    # A temperature around 0.8 is good for coherent text
    top_k=10            # Consider only the top 10 most likely next characters
)

# Decode the generated indices back to text
generated_text = decode(generated_indices[0].tolist())

print("\n--- Generated Text ---")
print(generated_text)