"""
Sample from a trained checkpoint. Output streams to stdout one character at a time.

    python generate.py "ROMEO:"
    python generate.py -n 1000 -t 0.8 -k 20 "JULIET:"
    echo "KING RICHARD III:" | python generate.py -
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from shakespeare_pytorch import DEFAULT_CKPT, load_checkpoint


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('prompt', nargs='?', default='\n', help="text to continue ('-' reads stdin; default: newline)")
    p.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT, help='checkpoint path (default: %(default)s)')
    p.add_argument('-n', '--max-new-tokens', type=int, default=500)
    p.add_argument('-t', '--temperature', type=float, default=1.0, help='<1 is safer, >1 is wilder')
    p.add_argument('-k', '--top-k', type=int, default=None, help='only sample from the k likeliest chars')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    if not args.ckpt.exists():
        print(f"error: no checkpoint at {args.ckpt}. Train one first: python shakespeare_pytorch.py",
              file=sys.stderr)
        return 1
    if args.temperature <= 0:
        print("error: --temperature must be > 0", file=sys.stderr)
        return 2

    prompt = sys.stdin.read() if args.prompt == '-' else args.prompt
    prompt = prompt or '\n'  # the model needs at least one token of context
    model, tok = load_checkpoint(args.ckpt, args.device)
    unknown = sorted(set(prompt) - set(tok.chars))
    if unknown:
        print(f"error: prompt has characters the model never saw: {unknown!r}", file=sys.stderr)
        return 2
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = next(model.parameters()).device
    context = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    sys.stdout.write(prompt)
    try:
        for t in model.generate(context, args.max_new_tokens, args.temperature, args.top_k):
            sys.stdout.write(tok.decode([t]))
            sys.stdout.flush()
        sys.stdout.write('\n')
    except KeyboardInterrupt:
        sys.stdout.write('\n')
        return 130
    except BrokenPipeError:  # e.g. piped into `head`; silence the interpreter's final flush
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


if __name__ == '__main__':
    sys.exit(main())
