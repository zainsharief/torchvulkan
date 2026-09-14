import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Deliberately hand-rolled with ``torch.bmm`` rather than
    ``F.scaled_dot_product_attention``: SDPA's own math fallback decomposes
    into the general broadcasting ``torch.matmul``, whose backward isn't
    wired up for every backend yet, while ``bmm``'s backward already is.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        # upper-triangular mask of future positions; a buffer so .to(device) moves it too
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def split_heads(self, t, B, T):
        return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2).reshape(B * self.n_head, T, self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = self.split_heads(q, B, T), self.split_heads(k, B, T), self.split_heads(v, B, T)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)

        out = out.view(B, self.n_head, T, self.head_dim).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """A minimal GPT-style decoder-only transformer.

    ``n_embd`` sets the model width (analogous to ``hidden`` in
    ``examples/mnist.py``); ``benchmark/benchmark.py llm`` sweeps it to scale
    the model up, holding ``n_layer`` fixed and deriving ``n_head`` from a
    constant per-head size.
    """

    def __init__(self, vocab_size: int = 1000, block_size: int = 64, n_layer: int = 4, n_embd: int = 256, head_dim: int = 64):
        super().__init__()
        assert n_embd % head_dim == 0
        n_head = n_embd // head_dim
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def main():
    import torch.optim as optim
    import torchvulkan as torchvk

    vocab_size = 32
    block_size = 16
    batch_size = 16
    learning_rate = 3e-3
    steps = 200
    device = torch.device('vulkan' if torchvk.is_available() else 'cpu')

    # No real corpus here (unlike mnist.py's MNIST download) -- this is a synthetic sanity
    # check: predict the next token of `x[i+1] = (x[i] + 1) % vocab_size`, a pattern trivial
    # for a transformer to learn, so a shrinking loss confirms the backend trains correctly
    # end to end (embedding -> attention -> mlp -> cross-entropy -> backward -> optimizer step).
    def make_batch():
        starts = torch.randint(0, vocab_size, (batch_size, 1))
        seq = (starts + torch.arange(block_size + 1)) % vocab_size
        return seq[:, :-1].to(device), seq[:, 1:].to(device)

    model = TinyGPT(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_embd=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for step in range(steps):
        data, targets = make_batch()
        logits = model(data)
        loss = criterion(logits.view(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\rstep [{step + 1}/{steps}] | loss: {loss.item():.4f}", end='')

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
