"""
Synthetic Turbofan Degradation Dataset Generator.
Generates realistic multi-unit engine degradation trajectories mimicking NASA C-MAPSS physics.
Used for immediate local testing, demonstration, and offline validation.
All generated records are explicitly marked as synthetic demonstration data.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from src.config import ALL_RAW_COLS, INDEX_COLS, SETTING_COLS, SENSOR_COLS, DEFAULT_RUL_CLIP


def generate_synthetic_turbofan_fleet(
    num_units: int = 50,
    min_cycles: int = 130,
    max_cycles: int = 340,
    rul_clip: Optional[int] = DEFAULT_RUL_CLIP,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate run-to-failure sensor data for a fleet of simulated turbofan aircraft engines.

    Degradation Physics Modeled:
    - Initial wear: Engines begin with slight manufacturing variations (healthy plateau).
    - Incubation phase: Stable operational readings with baseline Gaussian noise.
    - Exponential/polynomial damage propagation: Accelerated wear leading up to failure.
    - Sensor responses: Temperature and vibration increase; pressure and efficiency decrease.

    Args:
        num_units: Number of distinct engine units to generate.
        min_cycles: Minimum life cycle duration.
        max_cycles: Maximum life cycle duration.
        rul_clip: Upper threshold for piecewise-linear RUL clipping (None for unclipped).
        random_seed: Random state for deterministic repeatability.

    Returns:
        pd.DataFrame containing 26 C-MAPSS standard columns plus 'RUL' and 'is_synthetic'.
    """
    rng = np.random.default_rng(random_seed)
    records = []

    for unit_id in range(1, num_units + 1):
        # Total lifecycle for this specific unit
        total_life = int(rng.integers(min_cycles, max_cycles + 1))
        
        # Degradation onset cycle (incubation phase before noticeable wear)
        onset_cycle = int(rng.integers(int(total_life * 0.35), int(total_life * 0.65)))
        
        # Engine baseline variation
        base_t24 = 642.0 + rng.normal(0, 0.4)
        base_t30 = 1585.0 + rng.normal(0, 1.2)
        base_t50 = 1400.0 + rng.normal(0, 1.0)
        base_p30 = 554.0 + rng.normal(0, 0.5)
        base_nf = 2388.05 + rng.normal(0, 0.04)
        base_nc = 9050.0 + rng.normal(0, 2.0)
        base_ps30 = 47.45 + rng.normal(0, 0.08)
        base_phi = 521.8 + rng.normal(0, 0.3)
        base_nrf = 2388.08 + rng.normal(0, 0.03)
        base_nrc = 8130.0 + rng.normal(0, 1.5)
        base_bpr = 8.42 + rng.normal(0, 0.02)
        base_htbleed = 392.0 + rng.normal(0, 0.5)
        base_w31 = 38.85 + rng.normal(0, 0.05)
        base_w32 = 23.30 + rng.normal(0, 0.04)

        for cycle in range(1, total_life + 1):
            # Calculate degradation factor (0 when healthy, growing to 1 at failure)
            if cycle <= onset_cycle:
                deg_factor = 0.0
            else:
                # Quadratic damage propagation curve
                deg_factor = ((cycle - onset_cycle) / (total_life - onset_cycle)) ** 2.1

            # Target calculation (True remaining cycles to failure)
            linear_rul = total_life - cycle
            rul = min(rul_clip, linear_rul) if rul_clip is not None else linear_rul

            # Operational settings (single flight condition: Sea level standard)
            s_set1 = float(rng.normal(0.000, 0.0015))
            s_set2 = float(rng.normal(0.000, 0.0002))
            s_set3 = 100.0

            # Sensor readings with physics trends and measurement noise
            s1 = 518.67
            s2 = base_t24 + deg_factor * 1.8 + rng.normal(0, 0.15)
            s3 = base_t30 + deg_factor * 28.0 + rng.normal(0, 0.8)
            s4 = base_t50 + deg_factor * 32.0 + rng.normal(0, 0.9)
            s5 = 14.62
            s6 = 21.61
            s7 = base_p30 - deg_factor * 3.5 + rng.normal(0, 0.25)
            s8 = base_nf + deg_factor * 0.12 + rng.normal(0, 0.03)
            s9 = base_nc + deg_factor * 4.5 + rng.normal(0, 0.6)
            s10 = 1.30
            s11 = base_ps30 + deg_factor * 0.65 + rng.normal(0, 0.05)
            s12 = base_phi - deg_factor * 2.8 + rng.normal(0, 0.2)
            s13 = base_nrf + deg_factor * 0.10 + rng.normal(0, 0.03)
            s14 = base_nrc - deg_factor * 8.0 + rng.normal(0, 0.7)
            s15 = base_bpr + deg_factor * 0.14 + rng.normal(0, 0.015)
            s16 = 0.03
            s17 = base_htbleed + deg_factor * 5.0 + rng.normal(0, 0.4)
            s18 = 2388.0
            s19 = 100.0
            s20 = base_w31 - deg_factor * 0.45 + rng.normal(0, 0.04)
            s21 = base_w32 - deg_factor * 0.35 + rng.normal(0, 0.03)

            row = [
                unit_id, cycle, s_set1, s_set2, s_set3,
                s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21,
                rul
            ]
            records.append(row)

    cols = ALL_RAW_COLS + ["RUL"]
    df = pd.DataFrame(records, columns=cols)
    df["is_synthetic"] = True
    return df


def split_synthetic_train_test(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Split synthetic fleet into train (run-to-failure) and test (truncated trajectories).
    Maintains unit integrity without leakage across units.

    Returns:
        train_df: Full degradation trajectories for training units.
        test_df: Truncated operational trajectories for test units.
        test_rul: True RUL vector at the last observed cycle for each test unit.
    """
    rng = np.random.default_rng(random_seed)
    all_units = df["unit_nr"].unique()
    num_test = max(1, int(len(all_units) * test_ratio))
    test_units = rng.choice(all_units, size=num_test, replace=False)
    train_units = [u for u in all_units if u not in test_units]

    train_df = df[df["unit_nr"].isin(train_units)].copy().reset_index(drop=True)
    
    test_records = []
    test_final_rul = {}

    for u in sorted(test_units):
        u_df = df[df["unit_nr"] == u].sort_values("time_cycles")
        total_cycles = len(u_df)
        
        # Truncate at random point between 35% and 85% of total life
        cutoff = rng.integers(int(total_cycles * 0.35), int(total_cycles * 0.85))
        truncated_df = u_df.iloc[:cutoff].copy()
        test_records.append(truncated_df)
        
        # Ground truth RUL at the truncation point
        true_rul_at_cutoff = int(u_df.iloc[cutoff - 1]["RUL"])
        test_final_rul[u] = true_rul_at_cutoff

    test_df = pd.concat(test_records, ignore_index=True)
    rul_series = pd.Series(test_final_rul, name="RUL")
    rul_series.index.name = "unit_nr"

    return train_df, test_df, rul_series
