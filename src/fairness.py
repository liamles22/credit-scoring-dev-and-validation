"""Fairness proxy from area composition, as in docs/preregistration_fairness_zip3.md.

Group weights are constant within a ZIP3, so every statistic is built from per-area sums. That makes
the area bootstrap a reweighting of areas, not a resampling of loans.
"""
import numpy as np
import pandas as pd

GROUPS = ["White", "Black", "Hispanic"]
REFERENCE = "White"
ACS_COLUMNS = {"B03002_001E": "total", "B03002_003E": "White", "B03002_004E": "Black",
               "B03002_012E": "Hispanic", "B19013_001E": "median_income"}


def zip3_composition(acs: pd.DataFrame) -> pd.DataFrame:
    """Group shares, minority share and population-weighted median income per ZIP3.

    `acs` has a 5-digit `zcta` column and the raw ACS columns. Areas with no population are dropped.
    Median income is a population-weighted mean of ZCTA medians: context only.
    """
    a = acs.rename(columns=ACS_COLUMNS).copy()
    a["zip3"] = a["zcta"].astype(str).str.zfill(5).str[:3]
    income = a["median_income"].where(a["median_income"] > 0)  # ACS codes missing as large negatives
    a["income_x_pop"] = income * a["total"]
    a["pop_with_income"] = a["total"].where(income.notna(), 0)
    s = a.groupby("zip3")[["total", *GROUPS, "income_x_pop", "pop_with_income"]].sum()
    s = s[s["total"] > 0]
    out = s[GROUPS].div(s["total"], axis=0)
    out["minority_share"] = out["Black"] + out["Hispanic"]
    out["population"] = s["total"]
    out["median_income"] = s["income_x_pop"] / s["pop_with_income"].replace(0, np.nan)
    return out


def lc_zip3(zip_code: pd.Series) -> pd.Series:
    """Lending Club `190xx` -> `190`."""
    return zip_code.astype("string").str[:3]


def approve_lowest(score, n: int, seed: int = 0) -> np.ndarray:
    """Approve the `n` lowest scores; ties at the boundary are broken at random."""
    score = np.asarray(score, dtype=float)
    tiebreak = np.random.default_rng(seed).random(len(score))
    order = np.lexsort((tiebreak, score))
    approved = np.zeros(len(score), dtype=bool)
    approved[order[:n]] = True
    return approved


def area_sums(area: pd.Series, **flags) -> pd.DataFrame:
    """Loan count and the sum of each flag per area."""
    df = pd.DataFrame({"area": area.to_numpy(), "n": 1, **{k: np.asarray(v, float) for k, v in flags.items()}})
    return df.groupby("area").sum()


def group_rates(sums: pd.DataFrame, shares: pd.DataFrame, flag: str, area_weights=None) -> pd.Series:
    """`sum_i(w_ig * flag_i) / sum_i(w_ig)`, optionally with areas reweighted (bootstrap)."""
    k = np.ones(len(sums)) if area_weights is None else area_weights
    w = shares.loc[sums.index, GROUPS].to_numpy() * k[:, None]
    return pd.Series(w.T @ sums[flag].to_numpy() / (w.T @ sums["n"].to_numpy()), index=GROUPS)


def adverse_impact(rates: pd.Series) -> pd.Series:
    """Rate of each group divided by the reference group's rate."""
    return (rates / rates[REFERENCE]).drop(REFERENCE)


def bootstrap_area_weights(n_areas: int, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Multiplicity of each area in each resample of areas with replacement, shape (n_boot, n_areas)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_areas, np.full(n_areas, 1 / n_areas), size=n_boot).astype(float)


def area_quintiles(value: pd.Series, loans: pd.Series) -> pd.Series:
    """Quintile 1-5 per area, whole areas assigned so that each quintile holds about 20% of loans."""
    order = value.dropna().sort_values(kind="stable").index
    cum = loans.loc[order].cumsum() / loans.loc[order].sum()
    q = np.ceil(cum * 5 - 1e-9).clip(1, 5).astype(int)
    return pd.Series(q.to_numpy(), index=order)
