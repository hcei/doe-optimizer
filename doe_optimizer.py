"""DOE Bayesian Optimisation CLI.

Usage
-----
    python doe_optimizer.py                          # interactive mode (default)
    python doe_optimizer.py --simulate               # built-in simulation
    python doe_optimizer.py --budget 50 --seed 123   # custom settings
    python doe_optimizer.py --output history.csv     # export to CSV
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from process_simulator import ChemicalProcess
from bayesian_opt import BayesianOptimizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='DOE Bayesian Optimisation for industrial processes',
    )
    p.add_argument('--budget', type=int, default=30,
                   help='Maximum number of experiments (default: 30)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')
    p.add_argument('--simulate', action='store_true',
                   help='Run with built-in simulation mode')
    p.add_argument('--output', type=str, default=None,
                   help='Export optimisation history to CSV file')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress per-iteration output')
    return p.parse_args()


def print_header(budget: int, interactive: bool, seed: int):
    print('=' * 60)
    print('  DOE Bayesian Optimisation Solver')
    print('=' * 60)
    print(f'  Budget      : {budget} experiments')
    print(f'  Mode        : {"interactive" if interactive else "simulation"}')
    print(f'  Seed        : {seed}')
    print(f'  Factors     : temperature, pressure, time, catalyst')
    print('-' * 60)


def run_simulation(args: argparse.Namespace) -> None:
    """Run optimisation against the built-in chemical process simulator."""
    process = ChemicalProcess(seed=args.seed)
    bounds = np.array([
        process.FACTOR_BOUNDS[name] for name in process.FACTOR_NAMES
    ])

    # True optimum for reference
    true_params, true_yield = process.true_optimum()

    print_header(args.budget, False, args.seed)
    print(f'  True optimum yield (noiseless): {true_yield:.2f} %')
    print('-' * 60)

    optimizer = BayesianOptimizer(bounds, random_state=args.seed)

    for iteration in range(args.budget):
        x_next = optimizer.suggest()

        # Build params dict for simulator
        params = {name: float(x_next[i])
                  for i, name in enumerate(process.FACTOR_NAMES)}
        y = process.evaluate(params)

        optimizer.update(x_next, y)

        if not args.quiet:
            # 阶段标签：LHS（Latin Hypercube Sampling，初始探索阶段） / EI（Expected Improvement，模型驱动优化阶段）
            phase = 'LHS' if optimizer.is_initial_phase else 'EI '
            parts = ', '.join(
                f'{name}={params[name]:.2f}' for name in process.FACTOR_NAMES
            )
            print(f'  [{phase}] iter {iteration + 1:>3d}  |  '
                  f'{parts}  |  yield={y:.3f} %')

    # Final summary
    best_x = optimizer.best_params
    best_y = optimizer.best_value
    best_params = {name: float(best_x[i])
                   for i, name in enumerate(process.FACTOR_NAMES)}

    print('=' * 60)
    print('  OPTIMISATION COMPLETE')
    print('=' * 60)
    print(f'  Best observed yield : {best_y:.4f} %')
    print(f'  True optimum yield  : {true_yield:.4f} %')
    print(f'  Gap                 : {best_y - true_yield:+.4f} %')
    print(f'  Total experiments   : {optimizer.n_evals}')
    print()
    print('  Best parameters found:')
    for name in process.FACTOR_NAMES:
        lo, hi = process.FACTOR_BOUNDS[name]
        val = best_params[name]
        true_val = true_params[name]
        print(f'    {name:>12s}: {val:8.3f}  '
              f'(true: {true_val:8.3f},  range: {lo}-{hi})')

    # Export if requested
    if args.output:
        export_csv(optimizer, process, args.output)
        print(f'\n  History exported to: {args.output}')


def run_interactive(args: argparse.Namespace) -> None:
    """Run optimisation with user-provided experiment results."""
    # Define factor ranges
    factor_bounds = ChemicalProcess.FACTOR_BOUNDS
    factor_names = ChemicalProcess.FACTOR_NAMES
    bounds = np.array([factor_bounds[name] for name in factor_names])

    print_header(args.budget, True, args.seed)
    print('  Enter the measured yield (%) for each suggested experiment.')
    print('  Press Ctrl+C to abort.')
    print('-' * 60)

    optimizer = BayesianOptimizer(bounds, random_state=args.seed)

    for iteration in range(args.budget):
        x_next = optimizer.suggest()
        params = {name: float(x_next[i])
                  for i, name in enumerate(factor_names)}

        phase = 'LHS' if optimizer.is_initial_phase else 'EI '
        print()
        print(f'  [{phase}] Experiment {iteration + 1}/{args.budget}')
        print(f'  Suggested parameters:')
        for name in factor_names:
            print(f'    {name:>12s}: {params[name]:.3f}')

        # Read user result
        while True:
            try:
                raw = input('  Enter yield result (%) > ').strip()
                y = float(raw)
                break
            except ValueError:
                print('  [ERROR] Please enter a numeric value.')
            except (EOFError, KeyboardInterrupt):
                print('\n  Aborted by user.')
                sys.exit(0)

        optimizer.update(x_next, y)

    # Final summary
    if optimizer.n_evals > 0:
        best_x = optimizer.best_params
        best_y = optimizer.best_value
        best_params = {name: float(best_x[i])
                       for i, name in enumerate(factor_names)}

        print()
        print('=' * 60)
        print('  OPTIMISATION COMPLETE')
        print('=' * 60)
        print(f'  Best observed yield : {best_y:.4f} %')
        print(f'  Total experiments   : {optimizer.n_evals}')
        print()
        print('  Best parameters:')
        for name in factor_names:
            print(f'    {name:>12s}: {best_params[name]:.3f}')

    if args.output:
        export_csv(optimizer, None, args.output)
        print(f'\n  History exported to: {args.output}')


def export_csv(optimizer: BayesianOptimizer,
               process: ChemicalProcess | None,
               path: str) -> None:
    """Write optimisation history to CSV."""
    records = []
    for entry in optimizer.history:
        rec = {}
        for i, name in enumerate(ChemicalProcess.FACTOR_NAMES):
            rec[name] = entry['x'][i]
        rec['yield'] = entry['y']
        records.append(rec)

    df = pd.DataFrame(records)
    df.index.name = 'iteration'
    df.to_csv(path)


def main():
    args = parse_args()

    if args.simulate:
        run_simulation(args)
    else:
        run_interactive(args)


if __name__ == '__main__':
    main()