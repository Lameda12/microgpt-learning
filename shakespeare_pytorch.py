"""
Character-level GPT on Tiny Shakespeare, the PyTorch way.

Same algorithm as microgpt.py, but on tensors instead of scalar Values, so we can
afford a model ~50x larger. Train, then sample:

    python shakespeare_pytorch.py                  # trains, saves checkpoints/shakespeare.pt
    python generate.py "ROMEO:"                    # samples from the checkpoint
    python shakespeare_pytorch.py --help           # every knob
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_PATH = Path(__file__).with_name('shakespeare.txt')
DEFAULT_CKPT = Path(__file__).with_name('checkpoints') / 'shakespeare.pt'


# ---------------------------------------------------------------------------
# 1. Data and tokenizer

def load_text(path: Path = DATA_PATH) -> str:
    if not path.exists():
        print(f"downloading Tiny Shakespeare to {path}", file=sys.stderr)
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding='utf-8')


class CharTokenizer:
    """Every unique character in the corpus becomes a token id 0..vocab_size-1."""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.chars[i] for i in ids)


# ---------------------------------------------------------------------------
# 2. Model

@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64  # maximum context length
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """All heads at once: one matmul makes q, k, v for every head, then a fused attention kernel.

    Mathematically the same as a list of independent Head modules concatenated together,
    just without the Python loop over heads.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # softmax(q @ k^T / sqrt(head_size)) @ v, with future positions masked out
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all heads side by side
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """A simple two-layer MLP applied to every position independently."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (MLP)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)  # final layer norm
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = self.blocks(tok_emb + pos_emb)  # (B,T,C)
        logits = self.lm_head(self.ln_f(x))  # (B,T,vocab_size)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None):
        """Yield one sampled token id at a time, so callers can stream output."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float('-inf'))
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next.item()


# ---------------------------------------------------------------------------
# 3. Training

@dataclass
class TrainConfig:
    batch_size: int = 32
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 50  # batches used to estimate train loss (val loss is exact, see below)
    learning_rate: float = 1e-2  # 10x the original 1e-3: a 200K-param model at 1000 steps is badly undertrained
    lr_schedule: str = 'cosine'  # 'cosine' or 'constant'
    warmup_iters: int = 100
    min_lr_ratio: float = 0.1  # cosine decays to learning_rate * min_lr_ratio
    grad_clip: float = 1.0  # 0 disables clipping
    seed: int = 1337
    device: str = 'auto'


def resolve_device(name: str) -> str:
    if name != 'auto':
        return name
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_lr(it: int, tc: TrainConfig) -> float:
    """Linear warmup, then either hold (constant) or cosine-decay down to min_lr."""
    if it < tc.warmup_iters:
        return tc.learning_rate * (it + 1) / tc.warmup_iters
    if tc.lr_schedule == 'constant':
        return tc.learning_rate
    progress = (it - tc.warmup_iters) / max(1, tc.max_iters - tc.warmup_iters)
    min_lr = tc.learning_rate * tc.min_lr_ratio
    return min_lr + 0.5 * (tc.learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


@dataclass
class Data:
    tokenizer: CharTokenizer
    train: torch.Tensor
    val: torch.Tensor


def prepare_data(text: str | None = None, val_frac: float = 0.1) -> Data:
    text = load_text() if text is None else text
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int((1 - val_frac) * len(data))  # first 90% trains, last 10% validates
    return Data(tok, data[:n], data[n:])


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str,
              generator: torch.Generator | None = None):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model: GPT, data: Data, tc: TrainConfig, device: str) -> dict[str, float]:
    """Val loss over the *entire* val split (deterministic, so runs are comparable).
    Train loss over a fixed set of random batches (same batches every call)."""
    model.eval()
    bs = model.cfg.block_size
    n = (len(data.val) - 1) // bs
    xs = data.val[:n * bs].view(n, bs)
    ys = data.val[1:n * bs + 1].view(n, bs)
    total = 0.0
    for i in range(0, n, 256):
        _, loss = model(xs[i:i + 256].to(device), ys[i:i + 256].to(device))
        total += loss.item() * len(xs[i:i + 256])
    val = total / n

    g = torch.Generator().manual_seed(0)
    train = sum(
        model(*get_batch(data.train, tc.batch_size, bs, device, g))[1].item()
        for _ in range(tc.eval_iters)
    ) / tc.eval_iters
    model.train()
    return {'train': train, 'val': val}


def train(mc: GPTConfig, tc: TrainConfig, data: Data, log=print) -> tuple[GPT, dict]:
    """Train a fresh model. Returns the model and a history dict of eval points."""
    torch.manual_seed(tc.seed)
    device = resolve_device(tc.device)
    mc.vocab_size = data.tokenizer.vocab_size
    model = GPT(mc).to(device)
    log(f"device {device} | vocab {mc.vocab_size} | params {sum(p.numel() for p in model.parameters()):,}")

    # Instead of manual beta1, beta2 running averages like microgpt.py, PyTorch gives us Adam in 1 line
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate)
    history = {'step': [], 'train': [], 'val': [], 'time': []}
    t0 = time.perf_counter()
    train_time = 0.0  # excludes eval, so speedups are measured on training alone

    for it in range(tc.max_iters + 1):
        if it % tc.eval_interval == 0 or it == tc.max_iters:
            losses = evaluate(model, data, tc, device)
            for k, v in (('step', it), ('train', losses['train']), ('val', losses['val']), ('time', train_time)):
                history[k].append(v)
            log(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {train_time:6.1f}s")
        if it == tc.max_iters:
            break

        t_step = time.perf_counter()
        lr = get_lr(it, tc)
        for group in optimizer.param_groups:
            group['lr'] = lr
        xb, yb = get_batch(data.train, tc.batch_size, mc.block_size, device)
        _, loss = model(xb, yb)  # forward: build the graph up to the loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # backward: chain rule through the graph, exactly like Value.backward()
        if tc.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()  # update: nudge every parameter against its gradient
        if device == 'cuda':
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t_step

    history['wall_time'] = time.perf_counter() - t0
    return model, history


# ---------------------------------------------------------------------------
# 4. Checkpoints

def save_checkpoint(path: Path, model: GPT, tok: CharTokenizer, tc: TrainConfig, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_config': asdict(model.cfg),
        'train_config': asdict(tc),
        'chars': tok.chars,
        'state_dict': model.state_dict(),
        'history': history,
    }, path)


def load_checkpoint(path: Path, device: str = 'auto') -> tuple[GPT, CharTokenizer]:
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**ckpt['model_config']))
    model.load_state_dict(ckpt['state_dict'])
    model.to(resolve_device(device)).eval()
    return model, CharTokenizer(ckpt['chars'])


# ---------------------------------------------------------------------------
# 5. CLI

def add_config_args(p: argparse.ArgumentParser) -> None:
    """Expose every GPTConfig/TrainConfig field as --kebab-case flag."""
    for cls in (GPTConfig, TrainConfig):
        for name, default in asdict(cls()).items():
            if name == 'vocab_size':
                continue  # derived from the data
            flag = '--' + name.replace('_', '-')
            kw = {'choices': ['cosine', 'constant']} if name == 'lr_schedule' else {}
            p.add_argument(flag, type=type(default), default=default, metavar=name.upper(), **kw)


def configs_from_args(args: argparse.Namespace) -> tuple[GPTConfig, TrainConfig]:
    pick = lambda cls: cls(**{k: getattr(args, k) for k in asdict(cls()) if hasattr(args, k)})
    return pick(GPTConfig), pick(TrainConfig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument('--out', type=Path, default=DEFAULT_CKPT, help='checkpoint path (default: %(default)s)')
    p.add_argument('--no-save', action='store_true', help="don't write a checkpoint")
    p.add_argument('--sample-tokens', type=int, default=400, help='chars to sample after training (0 to skip)')
    args = p.parse_args()

    mc, tc = configs_from_args(args)
    data = prepare_data()
    model, history = train(mc, tc, data)

    if not args.no_save:
        save_checkpoint(args.out, model, data.tokenizer, tc, history)
        print(f"saved checkpoint to {args.out}")

    if args.sample_tokens > 0:
        print("\n--- inference (babbled Shakespeare) ---")
        context = torch.tensor([data.tokenizer.encode('\n')], device=next(model.parameters()).device)
        for t in model.generate(context, args.sample_tokens):
            print(data.tokenizer.decode([t]), end='', flush=True)
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
