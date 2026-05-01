#!/usr/bin/env python
# coding: utf-8

# ## Task 2

# ### pandas_grouped_stats.py

# In[9]:


import argparse
import json
from pathlib import Path

import pandas as pd

MISSING = ["", "na", "n/a", "null", "none", "nan", "missing", "-"]


def numeric_version(series):
    s = series.astype("string").str.strip()
    s = s.str.replace(r"[\$,()%]", "", regex=True).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def infer_type(series):
    non_missing = series.dropna().astype("string").str.strip()
    non_missing = non_missing[~non_missing.str.lower().isin(MISSING)]
    if len(non_missing) == 0:
        return "empty"
    numeric = numeric_version(non_missing)
    if numeric.notna().mean() >= 0.90:
        return "numeric"
    dates = pd.to_datetime(non_missing, errors="coerce")
    if dates.notna().mean() >= 0.70:
        return "date"
    return "categorical"


def numeric_stats(series):
    nums = numeric_version(series).dropna()
    if len(nums) == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None, "median": None}
    return {
        "count": int(nums.count()),
        "mean": round(float(nums.mean()), 6),
        "min": round(float(nums.min()), 6),
        "max": round(float(nums.max()), 6),
        "std": None if len(nums) < 2 else round(float(nums.std(ddof=1)), 6),
        "median": round(float(nums.median()), 6),
    }


def categorical_stats(series):
    s = series.astype("string").str.strip()
    s = s[~s.isna()]
    s = s[~s.str.lower().isin(MISSING)]
    vc = s.value_counts(dropna=True)
    if len(vc) == 0:
        return {"count": 0, "unique": 0, "mode": None, "mode_freq": 0, "top_5": []}
    return {
        "count": int(s.count()),
        "unique": int(s.nunique()),
        "mode": str(vc.index[0]),
        "mode_freq": int(vc.iloc[0]),
        "top_5": [(str(idx), int(val)) for idx, val in vc.head(5).items()],
    }


def auto_group_columns(df, inferred, max_groups=25):
    choices = []
    for col, typ in inferred.items():
        non_missing = df[col].dropna().astype("string").str.strip()
        unique = int(non_missing.nunique())
        if typ in {"categorical", "date"} and 2 <= unique <= max_groups:
            choices.append((unique, col))
    return [col for _, col in sorted(choices)[:3]]


def grouped_stats(df, group_cols, numeric_cols, cat_cols):
    output = {}
    for group_col in group_cols:
        output[group_col] = {}
        temp = df.copy()
        temp[group_col] = temp[group_col].fillna("(missing)").astype(str)

        for key, group in temp.groupby(group_col, dropna=False):
            output[group_col][str(key)] = {
                "row_count": int(len(group)),
                "numeric": {c: numeric_stats(group[c]) for c in numeric_cols},
                "categorical": {c: categorical_stats(group[c]) for c in cat_cols[:5] if c != group_col},
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="fb_ads_president_scored_anon.csv")
    parser.add_argument("--group-cols", default="", help="Comma-separated grouping columns. If omitted, script auto-selects.")
    parser.add_argument("--output", default="pandas_results.json")
    args, unknown = parser.parse_known_args()

    df = pd.read_csv(args.file, dtype="string", keep_default_na=False, na_values=MISSING)

    inferred = {c: infer_type(df[c]) for c in df.columns}
    numeric_cols = [c for c, t in inferred.items() if t == "numeric"]
    cat_cols = [c for c, t in inferred.items() if t in {"categorical", "date"}]

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    if not group_cols:
        group_cols = auto_group_columns(df, inferred)

    result = {
        "script": "pandas_grouped_stats.py",
        "dataset": args.file,
        "overall": {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "shape": list(df.shape),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        },
        "missing_values": {
            c: {
                "missing_count": int(df[c].isna().sum()),
                "missing_percent": round(float(df[c].isna().mean() * 100), 4),
            } for c in df.columns
        },
        "inferred_types": inferred,
        "numeric_summary": {c: numeric_stats(df[c]) for c in numeric_cols},
        "categorical_summary": {c: categorical_stats(df[c]) for c in cat_cols},
        "pandas_describe_numeric": df[[c for c in df.columns if c in numeric_cols]].apply(numeric_version).describe().round(6).to_dict() if numeric_cols else {},
        "pandas_describe_all": df.describe(include="all").fillna("").astype(str).to_dict(),
        "grouped_by": group_cols,
        "grouped_summary": grouped_stats(df, group_cols, numeric_cols, cat_cols),
    }

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


# ### polars_grouped_stats.py

# In[15]:


import argparse
import json
from collections import Counter
from pathlib import Path

import polars as pl

MISSING = {"", "na", "n/a", "null", "none", "nan", "missing", "-"}


def is_missing(value):
    return value is None or str(value).strip().lower() in MISSING


def clean_numeric_value(value):
    if is_missing(value):
        return None
    s = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def infer_type(values):
    vals = [v for v in values if not is_missing(v)]
    if not vals:
        return "empty"
    numeric_count = sum(clean_numeric_value(v) is not None for v in vals)
    if numeric_count / len(vals) >= 0.90:
        return "numeric"
    # Keep date detection simple and stable across Polars versions.
    date_like = sum(any(sep in str(v) for sep in ["-", "/"]) for v in vals)
    if date_like / len(vals) >= 0.70:
        return "date"
    return "categorical"


def numeric_stats(values):
    nums = [clean_numeric_value(v) for v in values if clean_numeric_value(v) is not None]
    if not nums:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None, "median": None}
    s = pl.Series(nums)
    return {
        "count": int(s.len()),
        "mean": round(float(s.mean()), 6),
        "min": round(float(s.min()), 6),
        "max": round(float(s.max()), 6),
        "std": None if s.len() < 2 else round(float(s.std(ddof=1)), 6),
        "median": round(float(s.median()), 6),
    }


def categorical_stats(values):
    vals = [str(v).strip() for v in values if not is_missing(v)]
    counts = Counter(vals)
    if not counts:
        return {"count": 0, "unique": 0, "mode": None, "mode_freq": 0, "top_5": []}
    mode, mode_freq = counts.most_common(1)[0]
    return {
        "count": len(vals),
        "unique": len(counts),
        "mode": mode,
        "mode_freq": mode_freq,
        "top_5": counts.most_common(5),
    }


def auto_group_columns(df, inferred, max_groups=25):
    choices = []
    for col, typ in inferred.items():
        vals = df[col].to_list()
        unique = len(set(v for v in vals if not is_missing(v)))
        if typ in {"categorical", "date"} and 2 <= unique <= max_groups:
            choices.append((unique, col))
    return [col for _, col in sorted(choices)[:3]]


def grouped_stats(df, group_cols, numeric_cols, cat_cols):
    output = {}
    for group_col in group_cols:
        output[group_col] = {}
        keys = df[group_col].fill_null("(missing)").cast(pl.Utf8).unique().to_list()
        for key in keys:
            group = df.filter(pl.col(group_col).fill_null("(missing)").cast(pl.Utf8) == str(key))
            output[group_col][str(key)] = {
                "row_count": int(group.height),
                "numeric": {c: numeric_stats(group[c].to_list()) for c in numeric_cols},
                "categorical": {c: categorical_stats(group[c].to_list()) for c in cat_cols[:5] if c != group_col},
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="fb_ads_president_scored_anon.csv")
    parser.add_argument("--group-cols", default="", help="Comma-separated grouping columns. If omitted, script auto-selects.")
    parser.add_argument("--output", default="polars_results.json")
    args, unknown = parser.parse_known_args()

    df = pl.read_csv(args.file, infer_schema_length=0, null_values=list(MISSING))

    columns = df.columns
    values_by_col = {c: df[c].to_list() for c in columns}
    inferred = {c: infer_type(values_by_col[c]) for c in columns}
    numeric_cols = [c for c, t in inferred.items() if t == "numeric"]
    cat_cols = [c for c, t in inferred.items() if t in {"categorical", "date"}]

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    if not group_cols:
        group_cols = auto_group_columns(df, inferred)

    result = {
        "script": "polars_grouped_stats.py",
        "dataset": args.file,
        "overall": {
            "total_rows": int(df.height),
            "total_columns": int(df.width),
            "columns": columns,
            "shape": [int(df.height), int(df.width)],
            "dtypes": {c: str(df.schema[c]) for c in columns},
        },
        "missing_values": {
            c: {
                "missing_count": int(df[c].null_count()),
                "missing_percent": round(float(df[c].null_count() / df.height * 100), 4) if df.height else 0,
            } for c in columns
        },
        "inferred_types": inferred,
        "numeric_summary": {c: numeric_stats(values_by_col[c]) for c in numeric_cols},
        "categorical_summary": {c: categorical_stats(values_by_col[c]) for c in cat_cols},
        "polars_describe": df.describe().to_dict(as_series=False),
        "grouped_by": group_cols,
        "grouped_summary": grouped_stats(df, group_cols, numeric_cols, cat_cols),
    }

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


# ### pure_python_grouped_stats.py

# In[20]:


import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

MISSING = {"", "na", "n/a", "null", "none", "nan", "missing", "-"}


def is_missing(value):
    return value is None or str(value).strip().lower() in MISSING


def clean_numeric(value):
    """Convert common money/percentage/comma-formatted strings to float."""
    if is_missing(value):
        return None
    s = str(value).strip()
    s = s.replace("$", "").replace(",", "").replace("%", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def infer_type(values):
    non_missing = [v for v in values if not is_missing(v)]
    if not non_missing:
        return "empty"
    numeric_count = sum(clean_numeric(v) is not None for v in non_missing)
    if numeric_count / len(non_missing) >= 0.90:
        return "numeric"
    date_count = 0
    for v in non_missing:
        s = str(v).strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%m/%d/%y"):
            try:
                datetime.strptime(s[:19], fmt)
                date_count += 1
                break
            except ValueError:
                pass
    if date_count / len(non_missing) >= 0.70:
        return "date"
    return "categorical"


def median(nums):
    nums = sorted(nums)
    n = len(nums)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2


def stdev_sample(nums):
    n = len(nums)
    if n < 2:
        return None
    mean = sum(nums) / n
    return math.sqrt(sum((x - mean) ** 2 for x in nums) / (n - 1))


def numeric_stats(values):
    nums = [clean_numeric(v) for v in values if clean_numeric(v) is not None]
    if not nums:
        return {
            "count": 0, "mean": None, "min": None, "max": None,
            "std": None, "median": None
        }
    return {
        "count": len(nums),
        "mean": round(sum(nums) / len(nums), 6),
        "min": round(min(nums), 6),
        "max": round(max(nums), 6),
        "std": None if stdev_sample(nums) is None else round(stdev_sample(nums), 6),
        "median": round(median(nums), 6),
    }


def categorical_stats(values):
    vals = [str(v).strip() for v in values if not is_missing(v)]
    counts = Counter(vals)
    mode, mode_freq = (None, 0) if not counts else counts.most_common(1)[0]
    return {
        "count": len(vals),
        "unique": len(counts),
        "mode": mode,
        "mode_freq": mode_freq,
        "top_5": counts.most_common(5),
    }


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []
    return rows, columns


def auto_group_columns(rows, columns, max_groups=25):
    """Choose up to 3 useful categorical columns for grouped summaries."""
    choices = []
    for col in columns:
        vals = [r.get(col, "") for r in rows if not is_missing(r.get(col, ""))]
        if not vals:
            continue
        dtype = infer_type(vals)
        unique = len(set(vals))
        if dtype in {"categorical", "date"} and 2 <= unique <= max_groups:
            choices.append((unique, col))
    return [col for _, col in sorted(choices)[:3]]


def grouped_stats(rows, group_cols, numeric_cols, cat_cols):
    output = {}
    for group_col in group_cols:
        groups = defaultdict(list)
        for row in rows:
            key = row.get(group_col, "")
            if is_missing(key):
                key = "(missing)"
            groups[str(key)].append(row)

        output[group_col] = {}
        for key, grows in groups.items():
            output[group_col][key] = {
                "row_count": len(grows),
                "numeric": {
                    c: numeric_stats([r.get(c, "") for r in grows])
                    for c in numeric_cols
                },
                "categorical": {
                    c: categorical_stats([r.get(c, "") for r in grows])
                    for c in cat_cols[:5] if c != group_col
                }
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="fb_ads_president_scored_anon.csv")
    parser.add_argument("--group-cols", default="", help="Comma-separated grouping columns. If omitted, script auto-selects.")
    parser.add_argument("--output", default="pure_python_results.json")
    args, unknown = parser.parse_known_args()

    rows, columns = read_csv(args.file)
    col_values = {c: [r.get(c, "") for r in rows] for c in columns}
    inferred = {c: infer_type(col_values[c]) for c in columns}
    numeric_cols = [c for c, t in inferred.items() if t == "numeric"]
    cat_cols = [c for c, t in inferred.items() if t in {"categorical", "date"}]

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    if not group_cols:
        group_cols = auto_group_columns(rows, columns)

    result = {
        "script": "pure_python_grouped_stats.py",
        "dataset": args.file,
        "overall": {
            "total_rows": len(rows),
            "total_columns": len(columns),
            "columns": columns,
        },
        "missing_values": {
            c: {
                "missing_count": sum(is_missing(v) for v in col_values[c]),
                "missing_percent": round((sum(is_missing(v) for v in col_values[c]) / len(rows) * 100), 4) if rows else 0,
            } for c in columns
        },
        "inferred_types": inferred,
        "numeric_summary": {c: numeric_stats(col_values[c]) for c in numeric_cols},
        "categorical_summary": {c: categorical_stats(col_values[c]) for c in cat_cols},
        "grouped_by": group_cols,
        "grouped_summary": grouped_stats(rows, group_cols, numeric_cols, cat_cols),
    }

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

