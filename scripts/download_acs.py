"""Download ACS 2013-2017 5-year estimates by ZCTA, as in docs/preregistration_fairness_zip3.md.

The Census API needs a free key (https://api.census.gov/data/key_signup.html). Put it in the
environment or in `.env` as CENSUS_API_KEY=...; it is never printed or saved with the data.

Run from the project root: python -m scripts.download_acs
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from src.data import PROJECT_ROOT
from src.fairness import ACS_COLUMNS

OUT = PROJECT_ROOT / "data" / "external" / "acs_zcta_2017.csv"
URL = "https://api.census.gov/data/2017/acs/acs5"


def api_key() -> str:
    key = os.environ.get("CENSUS_API_KEY")
    env = PROJECT_ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("CENSUS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise SystemExit("CENSUS_API_KEY is not set; see the module docstring.")
    return key


def fetch(geo: dict) -> pd.DataFrame:
    query = {"get": ",".join(ACS_COLUMNS), **geo, "key": api_key()}
    with urllib.request.urlopen(f"{URL}?{urllib.parse.urlencode(query)}", timeout=300) as r:
        rows = json.loads(r.read().decode("utf-8"))
    return pd.DataFrame(rows[1:], columns=rows[0])


def main():
    try:
        df = fetch({"for": "zip code tabulation area:*"})
    except urllib.error.HTTPError:
        # Before 2020 the API may nest ZCTAs in states.
        df = fetch({"for": "zip code tabulation area:*", "in": "state:*"})
    df = df.rename(columns={"zip code tabulation area": "zcta"})
    df[list(ACS_COLUMNS)] = df[list(ACS_COLUMNS)].apply(pd.to_numeric)
    df = df[["zcta", *ACS_COLUMNS]].drop_duplicates("zcta")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df):,} ZCTAs, population {df['B03002_001E'].sum():,.0f} -> {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
