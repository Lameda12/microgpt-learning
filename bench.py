"""
Benchmark a training config over several seeds and score it against the baseline.

The score is speedrun-style: how many training steps (and seconds) until val loss
reaches the baseline's final val loss. "30% better" = the target in <=70% of the steps.

    python bench.py --label baseline --lr-schedule constant --warmup-iters 0 --grad-clip 0
    python bench.py --label pr1                      # scored against bench/baseline.json
    python bench.py --label wide --n-embd 128        # any shakespeare_pytorch.py flag works

Results land in bench/<label>.json. Every run uses identical data and val set.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

from shakespeare_pytorch import add_config_args, configs_from_args, prepare_data, train

BENCH_DIR = Path(__file__).with_name('bench')


def steps_to_target(history: dict, target: float) -> tuple[int, float] | None:
    """First eval point where val loss <= target, as (step, train_seconds)."""
    for step, val, t in zip(history['step'], history['val'], history['time']):
        if val <= target:
            return step, t
    return None


def mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.set_defaults(eval_interval=25)  # finer resolution for steps-to-target
    p.add_argument('--label', required=True, help='name for this run; results go to bench/LABEL.json')
    p.add_argument('--seeds', type=int, nargs='+', default=[1337, 1338, 1339])
    p.add_argument('--target', type=float, default=None,
                   help='val loss to race to (default: final val mean from bench/baseline.json)')
    args = p.parse_args()

    baseline_path = BENCH_DIR / 'baseline.json'
    target = args.target
    if target is None and args.label != 'baseline':
        if not baseline_path.exists():
            print(f"error: {baseline_path} missing. Run the baseline first (see --help) or pass --target.",
                  file=sys.stderr)
            return 1
        target = json.loads(baseline_path.read_text())['summary']['final_val_mean']

    data = prepare_data()
    runs = []
    for seed in args.seeds:
        args.seed = seed
        mc, tc = configs_from_args(args)
        print(f"--- seed {seed} ---", file=sys.stderr)
        _, history = train(mc, tc, data, log=lambda s: print(s, file=sys.stderr))
        runs.append({'seed': seed, 'history': history})

    if target is None:  # we *are* the baseline: race to our own mean final loss
        target = statistics.fmean(r['history']['val'][-1] for r in runs)

    finals = [r['history']['val'][-1] for r in runs]
    times = [r['history']['time'][-1] for r in runs]
    hits = [steps_to_target(r['history'], target) for r in runs]
    reached = [h for h in hits if h is not None]
    summary = {
        'target_val': target,
        'final_val_mean': mean_std(finals)[0],
        'final_val_std': mean_std(finals)[1],
        'train_time_mean': mean_std(times)[0],
        'seeds_reaching_target': len(reached),
        'steps_to_target_mean': statistics.fmean(s for s, _ in reached) if reached else None,
        'time_to_target_mean': statistics.fmean(t for _, t in reached) if reached else None,
    }
    mc, tc = configs_from_args(args)
    out = {'label': args.label, 'model_config': asdict(mc), 'train_config': asdict(tc),
           'summary': summary, 'runs': runs}
    BENCH_DIR.mkdir(exist_ok=True)
    out_path = BENCH_DIR / f'{args.label}.json'
    out_path.write_text(json.dumps(out, indent=1))

    s = summary
    print(f"\n{args.label}: final val {s['final_val_mean']:.4f} ± {s['final_val_std']:.4f} "
          f"over {len(runs)} seeds | target {target:.4f}")
    if reached:
        print(f"  reached target in {s['steps_to_target_mean']:.0f} steps / {s['time_to_target_mean']:.1f}s "
              f"({len(reached)}/{len(runs)} seeds)")
        if args.target is None and args.label != 'baseline':  # target came from baseline.json
            base = json.loads(baseline_path.read_text())
            base_steps = base['train_config']['max_iters']
            print(f"  vs baseline: {1 - s['steps_to_target_mean'] / base_steps:.0%} fewer steps")
    else:
        print("  never reached target")
    print(f"  wrote {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
