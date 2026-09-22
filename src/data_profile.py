import pandas as pd

# Thresholds used to decide how to coarsely summarize a column.
UNIQUE_RATIO_THRESHOLD = 0.95   # cardinality / row_count at or above this -> identifier-like
MIN_ROWS_FOR_IDENTIFIER = 20    # below this many rows, the ratio above is too noisy to trust
LOW_CARDINALITY_MAX = 15        # cardinality at or below this -> report full value counts
TOP_N = 10                      # otherwise -> top N values + an "other" bucket


def _summarize_column(series: pd.Series, row_count: int, top_n: int = TOP_N) -> dict:
    """Type-appropriate coarse summary for one column.

    Classification order matters: cardinality-based checks (identifier, then
    low-cardinality) run before the numeric-dtype check, not after. A numeric
    dtype alone doesn't mean "distribution" -- an int64 `year` column with 6
    distinct values wants value counts, not a mean/std, and an int64 id column
    that's ~100% unique wants an identifier duplicate-count, not a mean/std.
    Numeric distribution stats only apply to numeric columns that survive both
    of those checks (i.e. genuinely many distinct, non-identifier values).

    - datetime -> min/max/span
    - near-unique (cardinality/row_count >= UNIQUE_RATIO_THRESHOLD, and enough
      rows to trust that ratio) -> identifier: how many values repeat, and how
      many rows that involves. Applies regardless of numeric/string dtype.
    - low cardinality (<= LOW_CARDINALITY_MAX distinct values) -> full value
      counts. Applies regardless of numeric/string dtype (catches year-like
      int columns as well as string categories).
    - numeric (remaining, moderate/high cardinality) -> min/25/50/75/max/mean/std
    - everything else -> top N value counts + an "other" bucket for the remainder
    """
    non_null = series.dropna()
    cardinality = int(non_null.nunique())

    if pd.api.types.is_datetime64_any_dtype(series):
        if non_null.empty:
            return {"kind": "datetime", "min": None, "max": None, "span_days": None}
        return {
            "kind": "datetime",
            "min": non_null.min(),
            "max": non_null.max(),
            "span_days": (non_null.max() - non_null.min()).days,
        }

    # Float columns are excluded here: continuous measurements (lab values,
    # prices) are naturally almost-always-unique, which isn't the same thing
    # as being an identifier -- they still want distribution stats below,
    # not a duplicate-value count.
    is_near_unique = (
        not pd.api.types.is_float_dtype(series)
        and row_count >= MIN_ROWS_FOR_IDENTIFIER
        and cardinality / row_count >= UNIQUE_RATIO_THRESHOLD
    )
    if is_near_unique:
        counts = non_null.value_counts()
        repeated = counts[counts > 1]
        return {
            "kind": "identifier",
            "duplicate_values": int(len(repeated)),
            "duplicate_rows": int((repeated - 1).sum()),
        }

    counts = non_null.value_counts()
    if cardinality <= LOW_CARDINALITY_MAX:
        return {"kind": "categorical", "counts": counts.to_dict()}

    if pd.api.types.is_numeric_dtype(series):
        if non_null.empty:
            return {"kind": "numeric", "min": None, "p25": None, "median": None,
                    "p75": None, "max": None, "mean": None, "std": None, "zero_pct": None}
        desc = non_null.describe()
        return {
            "kind": "numeric",
            "min": round(desc["min"], 2),
            "p25": round(desc["25%"], 2),
            "median": round(desc["50%"], 2),
            "p75": round(desc["75%"], 2),
            "max": round(desc["max"], 2),
            "mean": round(desc["mean"], 2),
            "std": round(desc["std"], 2),
            "zero_pct": round((non_null == 0).mean() * 100, 2),
        }

    top = counts.head(top_n)
    other = int(counts.iloc[top_n:].sum())
    top_counts = top.to_dict()
    if other:
        top_counts["other"] = other
    return {"kind": "categorical_top_n", "counts": top_counts}


def profile_dataframe(df: pd.DataFrame, sample_size: int = 5) -> dict:
    """Produce a quick data-quality profile of a dataframe.

    Returns a dict with:
      - shape: (n_rows, n_cols)
      - duplicate_rows: count of fully duplicated rows
      - columns: a DataFrame indexed by column name with dtype, null_pct,
        cardinality (nunique), a sample of distinct values, and a
        type-appropriate coarse summary (see _summarize_column).
    """
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())

    rows = []
    for col in df.columns:
        series = df[col]
        null_pct = round(series.isna().mean() * 100, 2)
        cardinality = int(series.nunique(dropna=True))
        samples = series.dropna().unique()[:sample_size]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "null_pct": null_pct,
                "cardinality": cardinality,
                "sample_values": list(samples),
                "summary": _summarize_column(series, n_rows),
            }
        )

    columns_df = pd.DataFrame(rows).set_index("column")

    return {
        "shape": (n_rows, n_cols),
        "duplicate_rows": duplicate_rows,
        "columns": columns_df,
    }


def inspect_column(df: pd.DataFrame, column: str, top_n: int = TOP_N, bins: int = 20):
    """Plot a closer look at one column you want to examine specifically.

    Uses the same classification as profile_dataframe, so a column shown as
    "categorical" there (e.g. a low-cardinality int `year` column) gets a bar
    chart here too, not a histogram just because its dtype happens to be numeric.
    Call this on individual columns that look interesting after reviewing
    profile_dataframe's coarse output, not on every column.
    """
    import matplotlib.pyplot as plt

    series = df[column].dropna()
    kind = _summarize_column(series, len(df), top_n=top_n)["kind"]

    if kind in ("numeric", "datetime"):
        series.hist(bins=bins)
        plt.title(f"{column} distribution")
        plt.xlabel(column)
        plt.ylabel("count")
    else:
        counts = series.value_counts().head(top_n)
        counts.plot(kind="bar")
        plt.title(f"{column} top {top_n} values")
        plt.ylabel("count")

    plt.tight_layout()
    return plt.gca()


def parse_messy_dates(series: pd.Series) -> pd.Series:
    """Parse a column of dates that may be a mix of DD/MM/YYYY strings,
    ISO 8601 strings (YYYY-MM-DD), and Excel serial numbers (days since
    1899-12-30, accounting for Excel's leap-year bug).

    Returns a pandas Series of datetime64[ns], with unparseable values as NaT.
    """
    excel_epoch = pd.Timestamp("1899-12-30")

    def parse_one(val):
        if pd.isna(val):
            return pd.NaT

        # Numeric types (or numeric-looking strings with no separators) are
        # treated as Excel serial dates.
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return excel_epoch + pd.to_timedelta(val, unit="D")

        s = str(val).strip()
        if not s:
            return pd.NaT

        if s.replace(".", "", 1).isdigit() and "/" not in s and "-" not in s:
            num = float(s)
            return excel_epoch + pd.to_timedelta(num, unit="D")

        # Try the two expected formats explicitly (avoids pandas guessing
        # and warning about ambiguity), then fall back to generic parsing
        # (dayfirst=True, UK/NHS convention) for anything else.
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return pd.to_datetime(s, format=fmt)
            except ValueError:
                continue
        return pd.to_datetime(s, dayfirst=True, errors="coerce")

    return series.apply(parse_one)
