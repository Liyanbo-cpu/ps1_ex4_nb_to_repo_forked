import pandas as pd

def combine_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Combine train and test data and add the 'set' column.

    Args:
    train_df: Train data
    test_df: Test data

    Returns:
    A combined DataFrame preserving the original indices.
    Rows with missing 'Survived' are labelled 'test' 
    and other rows are labelled 'train'.
    '''

    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df["Survived"].isna(), "set"] = "test"
    return all_df


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Return a copy with 'Family Size' calculated as SibSp + Parch + 1.

    Args:
    df: Passenger data containing 'SibSp' and 'Parch'.

    Returns:
    A new df containing the 'Family Size' column.
    The passenger is included in the family size.
    '''

    df = df.copy()
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df


def create_age_interval(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Return a copy with ages grouped into five intervals.

    Args:
    df: Passenger data containing the numeric 'Age' column.

    Returns:
    A new DataFrame containing 'Age Interval':
        0 for Age <= 16;
        1 for 16 < Age <= 32;
        2 for 32 < Age <= 48;
        3 for 48 < Age <= 64;
        4 for Age > 64.
    Missing ages remain 0.0.
    '''

    df = df.copy()
    df["Age Interval"] = 0.0

    df.loc[df["Age"] <= 16, "Age Interval"] = 0
    df.loc[
        (df["Age"] > 16) & (df["Age"] <= 32),
        "Age Interval",
    ] = 1
    df.loc[
        (df["Age"] > 32) & (df["Age"] <= 48),
        "Age Interval",
    ] = 2
    df.loc[
        (df["Age"] > 48) & (df["Age"] <= 64),
        "Age Interval",
    ] = 3
    df.loc[df["Age"] > 64, "Age Interval"] = 4

    return df


def create_fare_interval(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Return a copy with fares grouped into four intervals.

    Args:
    df: Passenger data containing the numeric 'Fare' column.

    Returns:
    A new DataFrame containing 'Fare Interval':
        0 for Fare <= 7.91;
        1 for 7.91 < Fare <= 14.454;
        2 for 14.454 < Fare <= 31;
        3 for Fare > 31.
    Missing fares remain 0.0.
    '''

    df = df.copy()
    df["Fare Interval"] = 0.0

    df.loc[df["Fare"] <= 7.91, "Fare Interval"] = 0
    df.loc[
        (df["Fare"] > 7.91) & (df["Fare"] <= 14.454),
        "Fare Interval",
    ] = 1
    df.loc[
        (df["Fare"] > 14.454) & (df["Fare"] <= 31),
        "Fare Interval",
    ] = 2
    df.loc[df["Fare"] > 31, "Fare Interval"] = 3

    return df


def create_sex_pclass(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Return a copy with a combined sex and passenger-class feature.

    Args:
    df: Passenger data containing 'Sex' as non-empty strings
        and 'Pclass' as passenger-class values.

    Returns:
    A new DataFrame containing 'Sex_Pclass', formed from the
    uppercase first character of 'Sex', '_C', and 'Pclass'.
    '''
    
    df = df.copy()
    df["Sex_Pclass"] = df.apply(
        lambda row: (
            row["Sex"][0].upper() + "_C" + str(row["Pclass"])
        ),
        axis=1,
    )
    return df