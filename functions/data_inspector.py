"""Summarise Titanic data without modifying the supplied DataFrame."""

from IPython.display import display

import pandas as pd


def inspect_data(df: pd.DataFrame) -> None:
    """Display the first five rows, structural info and summary statistics."""
    display(df.head())
    df.info()
    display(df.describe())


def calculate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts, percentages and types for each input column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect. The input is not modified.

    Returns
    -------
    pandas.DataFrame
        A transposed summary with input columns as columns and rows ``Total``,
        ``Percent`` and ``Types``. Percentages use all rows as the denominator
        and are undefined (NaN) when the dataset has no rows.
    """
    total = df.isna().sum()
    percent = df.isna().mean() * 100
    summary = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    summary["Types"] = df.dtypes.astype(str)
    return summary.T


def get_frequent_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return each column's most frequent non-missing value and its frequency.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect. The input is not modified.

    Returns
    -------
    pandas.DataFrame
        A transposed summary with rows ``Total`` (non-missing count),
        ``Most frequent item``, ``Frequence`` and ``Percent from total``.
        The original notebook's row labels are retained. Percentages exclude
        missing values and are rounded to three decimals. Ties follow pandas'
        ``value_counts()`` ordering. A column with no observed values has a
        missing item, zero frequency and an undefined (NaN) percentage.
    """
    summary = df.count().to_frame("Total")
    items = []
    frequencies = []
    for column in df.columns:
        counts = df[column].value_counts()
        counts = counts[counts > 0]
        items.append(counts.index[0] if not counts.empty else pd.NA)
        frequencies.append(counts.iloc[0] if not counts.empty else 0)
    summary["Most frequent item"] = items
    summary["Frequence"] = frequencies
    summary["Percent from total"] = (
        summary["Frequence"] / summary["Total"] * 100
    ).round(3)
    return summary.T


def get_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return non-missing counts and distinct-value counts for each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect. The input is not modified.

    Returns
    -------
    pandas.DataFrame
        A transposed summary with rows ``Total`` and ``Uniques``, preserving
        input column order. Both counts exclude missing values.
    """
    return pd.DataFrame({"Total": df.count(), "Uniques": df.nunique()}).T
