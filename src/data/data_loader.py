"""
src/data/data_loader.py

Data loading and preprocessing for CAS Schedule P loss triangles.

The raw CSVs are in long format — one row per (company, accident year,
development lag) cell. This module handles:
  - Loading and validating raw CSVs
  - Filtering to a single company or iterating over all companies
  - Pivoting to the familiar triangle grid
  - Splitting into upper (known) and lower (actual future) triangles
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All six lines available in the CAS Schedule P database
LINES = {
    "ppauto":  "Private Passenger Auto",
    "comauto": "Commercial Auto",
    "wkcomp":  "Workers Compensation",
    "medmal":  "Medical Malpractice",
    "othliab": "Other Liability",
    "prodliab": "Product Liability",
}

# Columns of interest (suffix varies by line — the loader handles this)
# AccidentYear  : year loss occurred        → triangle rows
# DevelopmentLag: years elapsed             → triangle columns
# CumPaidLoss_* : cumulative paid losses    → primary target
# IncurLoss_*   : incurred losses           → secondary target
# BulkLoss_*    : bulk + IBNR reserves

DATA_DIR = Path("data/cas_schedule_p/raw")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, prefix: str) -> str:
    """
    Find the actual column name for a given prefix.
    E.g. 'CumPaidLoss_' might appear as 'CumPaidLoss_B' in ppauto.csv.
    Raises ValueError if no match or ambiguous.
    """
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise ValueError(f"No column starting with '{prefix}' found. "
                         f"Available columns: {list(df.columns)}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous prefix '{prefix}': matches {matches}")
    return matches[0]


def _valuation_year(df: pd.DataFrame) -> int:
    """
    Infer the valuation year from the data.
    The upper triangle is everything where AccidentYear + DevelopmentLag - 1
    <= valuation year. CAS data is typically valued at the max calendar year
    present in the data.
    """
    return int((df["AccidentYear"] + df["DevelopmentLag"] - 1).max())


# ---------------------------------------------------------------------------
# Core loading
# ---------------------------------------------------------------------------

def load_line(line: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load the raw long-format CSV for one line of business.

    Parameters
    ----------
    line : str
        One of the keys in LINES, e.g. 'ppauto', 'wkcomp'.
    data_dir : Path
        Directory containing the raw CSV files.

    Returns
    -------
    pd.DataFrame
        Raw long-format dataframe with all companies.
    """
    if line not in LINES:
        raise ValueError(f"Unknown line '{line}'. Choose from: {list(LINES)}")

    # CAS has released files under two naming conventions:
    #   ppauto.csv  (older dataset)
    #   ppauto_pos.csv  (newer dataset, 2026 update)
    candidates = [
        data_dir / f"{line}_pos.csv",
        data_dir / f"{line}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"File not found. Tried: {[str(p) for p in candidates]}\n"
            f"Download from: https://www.casact.org/publications-research/"
            f"research/research-resources/loss-reserving-data-pulled-naic-schedule-p"
        )

    df = pd.read_csv(path)
    df["Line"] = line
    return df


def list_companies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of companies in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw long-format dataframe from load_line().

    Returns
    -------
    pd.DataFrame
        One row per company with GRCODE, GRNAME, and number of rows.
    """
    return (
        df.groupby(["GRCODE", "GRNAME"])
        .size()
        .reset_index(name="n_rows")
        .sort_values("GRCODE")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Triangle construction
# ---------------------------------------------------------------------------

def to_triangle(
    df: pd.DataFrame,
    grcode: int,
    value_col: str = "CumPaidLoss_",
) -> pd.DataFrame:
    """
    Pivot one company's long-format data into a triangle grid.

    Parameters
    ----------
    df : pd.DataFrame
        Raw long-format dataframe from load_line().
    grcode : int
        NAIC company code (GRCODE) to filter on.
    value_col : str
        Column prefix to use as cell values. Will be resolved automatically
        (e.g. 'CumPaidLoss_' → 'CumPaidLoss_B').

    Returns
    -------
    pd.DataFrame
        Triangle with AccidentYear as index, DevelopmentLag as columns.
        Upper triangle cells are populated; lower triangle cells are NaN
        by convention (even though the raw data includes them).
    """
    company = df[df["GRCODE"] == grcode].copy()
    if company.empty:
        raise ValueError(f"No data found for GRCODE={grcode}.")

    col = _resolve_column(company, value_col)

    triangle = company.pivot(
        index="AccidentYear",
        columns="DevelopmentLag",
        values=col,
    )
    triangle.index.name = "AccidentYear"
    triangle.columns.name = "DevelopmentLag"

    return triangle


def split_triangle(triangle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a full square into upper and lower triangles.

    The valuation year is inferred as the maximum calendar year
    (AccidentYear + DevelopmentLag - 1) in the triangle.

    Upper triangle : calendar year <= valuation year  → known at valuation date
    Lower triangle : calendar year >  valuation year  → actual future development

    Parameters
    ----------
    triangle : pd.DataFrame
        Full square from to_triangle(), before any masking.

    Returns
    -------
    upper : pd.DataFrame  (NaN in lower-right cells)
    lower : pd.DataFrame  (NaN in upper-left cells)
    """
    upper = triangle.copy().astype(float)
    lower = triangle.copy().astype(float)

    valuation_year = int(triangle.index.max())  # max accident year = valuation year
    # More precisely: max calendar year = max(AccidentYear) + max(DevLag) - 1
    # but CAS data is square so this is equivalent.

    for ay in triangle.index:
        for lag in triangle.columns:
            cal_year = ay + lag - 1
            if cal_year > valuation_year:
                upper.loc[ay, lag] = np.nan   # mask future in upper
            else:
                lower.loc[ay, lag] = np.nan   # mask past in lower

    return upper, lower


# ---------------------------------------------------------------------------
# Convenience: load everything for one company
# ---------------------------------------------------------------------------

def load_company(
    line: str,
    grcode: int,
    value_col: str = "CumPaidLoss_",
    data_dir: Path = DATA_DIR,
) -> dict:
    """
    One-shot loader: returns a dict with the raw rows, full square,
    upper triangle, and lower triangle for a single company.

    Parameters
    ----------
    line    : str  — line of business, e.g. 'ppauto'
    grcode  : int  — NAIC company code
    value_col : str — column prefix, e.g. 'CumPaidLoss_' or 'IncurLoss_'
    data_dir  : Path

    Returns
    -------
    dict with keys: 'raw', 'square', 'upper', 'lower', 'grname', 'line'

    Example
    -------
    >>> result = load_company("ppauto", grcode=43)
    >>> result["upper"]   # upper triangle (training data)
    >>> result["lower"]   # lower triangle (ground truth)
    """
    df = load_line(line, data_dir)

    grname_rows = df[df["GRCODE"] == grcode]["GRNAME"]
    if grname_rows.empty:
        raise ValueError(f"GRCODE={grcode} not found in line '{line}'.")
    grname = grname_rows.iloc[0]

    square = to_triangle(df, grcode, value_col)
    upper, lower = split_triangle(square)

    return {
        "line":   line,
        "grcode": grcode,
        "grname": grname,
        "raw":    df[df["GRCODE"] == grcode].reset_index(drop=True),
        "square": square,
        "upper":  upper,
        "lower":  lower,
    }


# ---------------------------------------------------------------------------
# Convenience: load all companies for a line
# ---------------------------------------------------------------------------

def load_all_companies(
    line: str,
    value_col: str = "CumPaidLoss_",
    data_dir: Path = DATA_DIR,
    min_lags: int = 10,
) -> dict[int, dict]:
    """
    Load triangles for every company in a line of business.

    Parameters
    ----------
    line      : str  — line of business
    value_col : str  — column prefix
    data_dir  : Path
    min_lags  : int  — skip companies with fewer than this many development lags

    Returns
    -------
    dict mapping GRCODE -> result dict (same structure as load_company)
    """
    df = load_line(line, data_dir)
    grcodes = df["GRCODE"].unique()

    results = {}
    skipped = 0

    for grcode in grcodes:
        company_df = df[df["GRCODE"] == grcode]
        n_lags = company_df["DevelopmentLag"].nunique()
        if n_lags < min_lags:
            skipped += 1
            continue
        try:
            square = to_triangle(df, grcode, value_col)
            upper, lower = split_triangle(square)
            grname = company_df["GRNAME"].iloc[0]
            results[grcode] = {
                "line":   line,
                "grcode": grcode,
                "grname": grname,
                "square": square,
                "upper":  upper,
                "lower":  lower,
            }
        except Exception:
            skipped += 1
            continue

    print(f"Loaded {len(results)} companies for '{line}' "
          f"(skipped {skipped} with < {min_lags} lags).")
    return results


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    line = sys.argv[1] if len(sys.argv) > 1 else "ppauto"
    print(f"\nLoading line: {line}")

    df = load_line(line)
    print(f"  Rows: {len(df):,}")
    print(f"  Companies: {df['GRCODE'].nunique()}")
    print(f"  Columns: {list(df.columns)}")

    companies = list_companies(df)
    print(f"\nFirst 5 companies:\n{companies.head()}")

    grcode = int(companies['GRCODE'].iloc[0])
    result = load_company(line, grcode)
    print(f"\nCompany: {result['grname']} (GRCODE={grcode})")
    print(f"\nUpper triangle:\n{result['upper']}")
    print(f"\nLower triangle:\n{result['lower']}")
