import pandas as pd


def parse_names(row):
    """
    Extract the family name, title, given name, and maiden name
    from a Titanic passenger record.

    Parameters
    ----------
    row : pandas.Series
        A passenger record containing a 'Name' field.

    Returns
    -------
    pandas.Series or None
        Family name, title, given name, and maiden name.
        Returns None if parsing fails.
    """
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]

        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]

        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")
def parse_names_column(df):
    """
    Extract name components for all passengers in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Passenger data containing a "Name" column.

    Returns
    -------
    pandas.DataFrame
        A copy of the input data with four additional columns:
        Family Name, Title, Given Name, and Maiden Name.
    """
    result = df.copy()
    name_columns = ["Family Name", "Title", "Given Name", "Maiden Name"]
    result[name_columns] = result.apply(parse_names, axis=1)
    return result


def aggregate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group passenger titles into common categories.

    Parameters
    ----------
    df : pandas.DataFrame
        Passenger data containing a "Title" column.

    Returns
    -------
    pandas.DataFrame
        A copy of the data with an additional "Titles" column.
        Titles are grouped into Mr., Mrs., Miss., Master., and Rare.
    """
    result = df.copy()

    result["Titles"] = result["Title"].replace({
        "Mlle.": "Miss.",
        "Ms.": "Miss.",
        "Mme.": "Mrs."
    })

    rare_titles = [
        "Lady.", "the Countess.", "Capt.", "Col.",
        "Don.", "Dr.", "Major.", "Rev.",
        "Sir.", "Jonkheer.", "Dona."
    ]

    result["Titles"] = result["Titles"].replace(rare_titles, "Rare")

    return result


def show_title_survival_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate survival rates by passenger title and sex.

    Parameters
    ----------
    df : pandas.DataFrame
        Training data containing "Titles", "Sex", and "Survived".

    Returns
    -------
    pandas.DataFrame
        A table with Titles, Sex, and the mean survival rate
        in the Survived column (between 0 and 1).
    """
    result = (
        df[["Titles", "Sex", "Survived"]]
        .groupby(["Titles", "Sex"], as_index=False)["Survived"]
        .mean()
    )

    return result