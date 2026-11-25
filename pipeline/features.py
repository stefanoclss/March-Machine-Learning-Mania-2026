"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: synth:square:diff_srs_margin, synth:square:prior_srs_prob, synth:log1p:diff_log_seed
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('synth:square:diff_srs_margin', 'synth:square:prior_srs_prob', 'synth:log1p:diff_log_seed')
PRIORITY_SIGNAL_COLUMNS: list[str] = ['prior_srs_prob', 'prior_elo_prob', 'diff_log_seed', 'diff_srs_margin', 'diff_elo_max', 'diff_quality_win_score']


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit, human-readable engineered features one transformation per block."""
    out = df.copy()
    if not ACTIVE_FEATURE_OPS:
        return out

    fallback_cols = [c for c in PRIORITY_SIGNAL_COLUMNS if c in out.columns]
    if len(fallback_cols) < 2:
        for col in out.columns:
            if col not in fallback_cols and not str(col).endswith(("_isna", "_target_enc")):
                fallback_cols.append(str(col))
            if len(fallback_cols) >= 4:
                break

    if 'diff_srs_margin' in out.columns:
        out['diff_srs_margin_sq'] = np.square(out['diff_srs_margin']).clip(0.0, 25.0)
    if 'prior_srs_prob' in out.columns:
        out['prior_srs_prob_sq'] = np.square(out['prior_srs_prob']).clip(0.0, 25.0)
    if 'diff_log_seed' in out.columns:
        out['diff_log_seed_log1p'] = np.sign(out['diff_log_seed']) * np.log1p(np.abs(out['diff_log_seed']))

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
