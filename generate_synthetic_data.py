"""
Synthetic Data Generation CLI Script.
Generates realistic turbofan engine degradation trajectories mimicking NASA C-MAPSS physics.
Output is saved to data/synthetic_engine_fleet.csv or a custom path with metadata flags.
"""

import argparse
from pathlib import Path
from src.config import DATA_DIR, DEFAULT_RUL_CLIP
from src.synthetic_generator import generate_synthetic_turbofan_fleet


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic aircraft engine degradation dataset.")
    parser.add_argument("--units", type=int, default=50, help="Number of simulated engine units (default: 50)")
    parser.add_argument("--min-cycles", type=int, default=140, help="Minimum engine lifetime (default: 140)")
    parser.add_argument("--max-cycles", type=int, default=320, help="Maximum engine lifetime (default: 320)")
    parser.add_argument("--rul-clip", type=int, default=DEFAULT_RUL_CLIP, help="RUL piece-wise clipping threshold (default: 125)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "synthetic_engine_fleet.csv"), help="Output CSV path")

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AIRBUS PREDICTIVE MAINTENANCE SYSTEM - SYNTHETIC DATA GENERATOR")
    print("=" * 70)
    print(f"Generating {args.units} synthetic engine trajectories (Cycles: {args.min_cycles}-{args.max_cycles})...")
    print(f"RUL clipping threshold: {args.rul_clip}")

    df = generate_synthetic_turbofan_fleet(
        num_units=args.units,
        min_cycles=args.min_cycles,
        max_cycles=args.max_cycles,
        rul_clip=args.rul_clip,
        random_seed=args.seed
    )

    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df):,} operational records across {args.units} units.")
    print(f"Saved to: {output_path}")
    print("NOTE: All records are tagged with 'is_synthetic = True' to ensure audit transparency.")


if __name__ == "__main__":
    main()
