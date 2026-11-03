"""ZIP3 fairness proxy on OOT 2017, as in docs/preregistration_fairness_zip3.md. Runs once.

Needs data/external/acs_zcta_2017.csv from scripts.download_acs. Refuses to run if the result
table exists, so outcomes cannot be recomputed silently.

Run from the project root: python -m scripts.fairness_zip3
"""
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, TARGET
from src.fairness import (GROUPS, adverse_impact, approve_lowest, area_quintiles, area_sums,
                          bootstrap_area_weights, group_rates, lc_zip3, zip3_composition)
from src.features_config import CATEGORICAL
from src.scoring import Models
from src.split import PROCESSED_PATH

ACS = PROJECT_ROOT / "data" / "external" / "acs_zcta_2017.csv"
TABLES = PROJECT_ROOT / "reports" / "tables"
RESULT = TABLES / "fairness_zip3_air.csv"
APPROVAL_SHARE = 0.85
AMBER, RED, CALIBRATION_GAP, GRADE_MARGIN = 0.9, 0.8, 0.10, 0.02
N_BOOT, SEED = 1000, 0
CHAMPION = "LightGBM monotone"


def interval(x):
    return np.percentile(x, [2.5, 97.5])


def main():
    if RESULT.exists():
        raise SystemExit(f"{RESULT.name} exists: this analysis runs once (see the pre-registration).")
    shares = zip3_composition(pd.read_csv(ACS, dtype={"zcta": str}))

    d = pd.read_parquet(PROCESSED_PATH, filters=[("sample", "in", ["train", "oot"])])
    cats = {c: sorted(d.loc[d["sample"] == "train", c].dropna().astype(str).unique()) for c in CATEGORICAL}
    models = Models(cats)
    train, oot = d[d["sample"] == "train"], d[d["sample"] == "oot"].reset_index(drop=True)
    cutoff = np.quantile(models.predict(train)[CHAMPION], APPROVAL_SHARE)
    pd_oot = models.predict(oot)
    n_approved = int((pd_oot[CHAMPION] <= cutoff).sum())

    oot["zip3"] = lc_zip3(oot["zip_code"])
    matched = oot["zip3"].isin(shares.index).to_numpy()
    print(f"OOT loans {len(oot):,}; matched to census {matched.mean():.4f}; "
          f"approval share {n_approved / len(oot):.4f}", flush=True)

    sub_grade = oot["sub_grade"].astype(str).rank(method="dense").to_numpy()
    approved = {CHAMPION: approve_lowest(pd_oot[CHAMPION], n_approved, SEED),
                "Scorecard": approve_lowest(pd_oot["Scorecard"], n_approved, SEED),
                "Sub-grade": approve_lowest(sub_grade, n_approved, SEED)}
    m = oot[matched]
    y = m[TARGET].to_numpy()
    flags = {f"a_{k}": v[matched] for k, v in approved.items()}
    flags.update({f"bad_a_{k}": v[matched] * y for k, v in approved.items()})
    flags.update({"y": y, "p_champion": pd_oot[CHAMPION][matched], "p_scorecard": pd_oot["Scorecard"][matched]})
    sums = area_sums(m["zip3"], **flags)
    boot = bootstrap_area_weights(len(sums), N_BOOT, SEED)

    # 1 and 3: approval rates, AIR and the difference against the grade
    rows, air_boot = [], {}
    for name in approved:
        rates = group_rates(sums, shares, f"a_{name}")
        air = adverse_impact(rates)
        air_boot[name] = np.array([adverse_impact(group_rates(sums, shares, f"a_{name}", k)).to_numpy()
                                   for k in boot])
        bad = group_rates(sums, shares, f"bad_a_{name}") / rates
        for i, g in enumerate(air.index):
            lo, hi = interval(air_boot[name][:, i])
            rows.append({"score": name, "group": g, "approval_rate": rates[g], "white_rate": rates["White"],
                         "air": air[g], "air_lo": lo, "air_hi": hi,
                         "default_rate_approved": bad[g], "white_default_rate_approved": bad["White"]})
    air_table = pd.DataFrame(rows)
    diff = []
    for i, g in enumerate(["Black", "Hispanic"]):
        point = (air_table.query("score == @CHAMPION and group == @g")["air"].item()
                 - air_table.query("score == 'Sub-grade' and group == @g")["air"].item())
        lo, hi = interval(air_boot[CHAMPION][:, i] - air_boot["Sub-grade"][:, i])
        diff.append({"group": g, "champion_minus_subgrade": point, "lo": lo, "hi": hi})
    diff = pd.DataFrame(diff)

    # 2: calibration by minority-share quintile
    q = area_quintiles(shares.loc[sums.index, "minority_share"], sums["n"])
    by_q = sums.groupby(q).sum()
    calib = pd.DataFrame({"loans": by_q["n"], "minority_share_mean":
                          (shares.loc[sums.index, "minority_share"] * sums["n"]).groupby(q).sum() / by_q["n"],
                          "observed": by_q["y"] / by_q["n"],
                          "obs_pred_champion": by_q["y"] / by_q["p_champion"],
                          "obs_pred_scorecard": by_q["y"] / by_q["p_scorecard"]})
    calib.index.name = "minority_quintile"
    gap_boot = []
    for k in boot:
        b = (sums[["y", "p_champion"]].mul(k, axis=0)).groupby(q).sum()
        r = b["y"] / b["p_champion"]
        gap_boot.append(r.loc[1] - r.loc[5])
    gap = calib.loc[1, "obs_pred_champion"] - calib.loc[5, "obs_pred_champion"]
    gap_lo, gap_hi = interval(gap_boot)

    # Context, no rule: AIR by income quintile
    iq = area_quintiles(shares.loc[sums.index, "median_income"], sums["n"])
    income = []
    for quint in sorted(iq.unique()):
        idx = iq.index[iq == quint]
        air = adverse_impact(group_rates(sums.loc[idx], shares, f"a_{CHAMPION}"))
        income.append({"income_quintile": quint, "loans": int(sums.loc[idx, "n"].sum()), **air.to_dict()})
    income = pd.DataFrame(income)

    TABLES.mkdir(parents=True, exist_ok=True)
    diff.round(5).to_csv(TABLES / "fairness_zip3_vs_grade.csv", index=False)
    calib.round(5).to_csv(TABLES / "fairness_zip3_calibration.csv")
    income.round(5).to_csv(TABLES / "fairness_zip3_income.csv", index=False)
    air_table.round(5).to_csv(RESULT, index=False)

    print(air_table.round(4).to_string(index=False))
    print(diff.round(4).to_string(index=False))
    print(calib.round(4).to_string())
    print(income.round(4).to_string(index=False))
    champ = air_table[air_table["score"] == CHAMPION].set_index("group")["air"]
    level = "Red" if (champ < RED).any() else "Amber" if (champ < AMBER).any() else "Green"
    print(f"Rule 1, approval: {level} (min AIR {champ.min():.4f})")
    print(f"Rule 2, calibration gap Q1 - Q5: {gap:.4f} (95% CI {gap_lo:.4f} to {gap_hi:.4f}); "
          f"overstates risk in high-minority areas: {gap > CALIBRATION_GAP}; "
          f"understates: {gap < -CALIBRATION_GAP}")
    print(f"Rule 3, adds disparity over the grade: {bool((diff['champion_minus_subgrade'] < -GRADE_MARGIN).any())}")


if __name__ == "__main__":
    main()
