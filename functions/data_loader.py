"""Load the CSV files used in the Titanic analysis."""

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame using pandas' default parsing options.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file location. Relative paths are resolved from the working directory.

    Returns
    -------
    pandas.DataFrame
        Loaded data, with column names and inferred types preserved.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist. CSV parsing errors also propagate
        to the caller so that invalid input is not silently ignored.
    """
    return pd.read_csv(path)
