import pandas as pd


def create_family_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify passengers according to their family size.

    Parameters
    ----------
    df : pandas.DataFrame
        Passenger data containing a "Family Size" column.

    Returns
    -------
    pandas.DataFrame
        A copy of the input data with a "Family Type" column:
        Single (size 1), Small (size 2-4), or Large (size >= 5).
    """
    result = df.copy()

    result["Family Type"] = result["Family Size"].astype(object)

    result.loc[result["Family Size"] == 1, "Family Type"] = "Single"
    result.loc[
        (result["Family Size"] > 1) & (result["Family Size"] < 5),
        "Family Type"
    ] = "Small"
    result.loc[result["Family Size"] >= 5, "Family Type"] = "Large"

    return result