import pandas as pd

# Thresholds used to decide how to coarsely summarize a column.
UNIQUE_RATIO_THRESHOLD = 0.95   # cardinality / row_count at or above this -> identifier-like
MIN_ROWS_FOR_IDENTIFIER = 20    # below this many rows, the ratio above is too noisy to trust
LOW_CARDINALITY_MAX = 15        # cardinality at or below this -> report full value counts
TOP_N = 10                      # otherwise -> top N values + an "other" bucket

# Tokens these government data tables use in place of a number when a value
# is suppressed (e.g. small-number disclosure control) or intentionally not
# published. A non-numeric value is only ever treated as a placeholder, not
# genuine categorical data, when it exactly matches one of these -- anything
# else (e.g. an age band like "45 and over") is left as a real category.
SPECIAL_VALUE_TOKENS = {"not included", "z", ":", "*", "c", "n/a", "-", "low"}


def _numeric_or_none(value: str):
    """Parse a stripped string as a float, or return None if it isn't one."""
    try:
        return float(value)
    except ValueError:
        return None


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
    - numeric_w_placeholders (object dtype, at least one genuinely numeric
      value, and every non-numeric value is a known placeholder like "not
      included" or "z" -- see SPECIAL_VALUE_TOKENS) -> distribution stats over
      the parseable values plus a breakdown of the placeholder rows. Checked
      before the identifier/categorical rules below so these columns (read as
      object dtype by pandas because of the placeholder text) get treated as
      the numeric measures they are rather than as identifiers or categories.
      A column with unrecognized non-numeric values (e.g. an age band like
      "45 and over") does NOT match this and falls through to those rules.
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

    if not pd.api.types.is_numeric_dtype(series) and not non_null.empty:
        stripped = non_null.astype(str).str.strip()
        parsed = stripped.map(_numeric_or_none)
        is_special_token = stripped.str.lower().isin(SPECIAL_VALUE_TOKENS)
        unparsed = parsed.isna()
        # Every non-numeric value must be a recognized placeholder, and there
        # must be at least one real number -- otherwise this is either a
        # genuine category column or entirely non-numeric text.
        if parsed.notna().any() and (unparsed == is_special_token).all():
            numeric_values = parsed.dropna()
            special_counts = stripped[unparsed].value_counts().to_dict()
            desc = numeric_values.describe()
            return {
                "kind": "numeric_w_placeholders",
                "min": round(desc["min"], 2),
                "p25": round(desc["25%"], 2),
                "median": round(desc["50%"], 2),
                "p75": round(desc["75%"], 2),
                "max": round(desc["max"], 2),
                "mean": round(desc["mean"], 2),
                "std": round(desc["std"], 2),
                "special_rows": int(unparsed.sum()),
                "special_values": special_counts,
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
        valid_pct (rows that are neither null nor a placeholder token -- i.e.
        actually usable as the column's real datatype; equals 100 - null_pct
        except on numeric_w_placeholders columns, where placeholder rows are
        also subtracted), cardinality (nunique), a sample of distinct values,
        a type-appropriate coarse summary (see _summarize_column), and
        special_values -- the placeholder-token breakdown for
        numeric_w_placeholders columns (pulled out of summary for
        visibility), None for every other column.
    """
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())

    rows = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round(null_count / n_rows * 100, 2) if n_rows else None
        cardinality = int(series.nunique(dropna=True))
        samples = series.dropna().unique()[:sample_size]
        summary = _summarize_column(series, n_rows)
        is_placeholder_col = summary["kind"] == "numeric_w_placeholders"
        special_rows = summary["special_rows"] if is_placeholder_col else 0
        valid_pct = round((n_rows - null_count - special_rows) / n_rows * 100, 2) if n_rows else None
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "null_pct": null_pct,
                "valid_pct": valid_pct,
                "cardinality": cardinality,
                "sample_values": list(samples),
                "summary": summary,
                "special_values": summary.get("special_values") if is_placeholder_col else None,
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

    if kind == "numeric_w_placeholders":
        numeric_series = series.astype(str).str.strip().map(_numeric_or_none).dropna()
        numeric_series.hist(bins=bins)
        plt.title(f"{column} distribution (excludes placeholder rows)")
        plt.xlabel(column)
        plt.ylabel("count")
    elif kind in ("numeric", "datetime"):
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
