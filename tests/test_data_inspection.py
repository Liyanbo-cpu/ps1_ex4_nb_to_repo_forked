"""Regression tests for the notebook's data loading and inspection helpers."""

import io
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from functions.data_inspector import (
    calculate_missing_values,
    get_frequent_data,
    get_unique_values,
    inspect_data,
)
from functions.data_loader import load_data


ROOT = Path(__file__).resolve().parents[1]


def notebook_missing_reference(df):
    """Reproduce notebook cells 17/18 without calling the refactored helper."""
    total = df.isnull().sum()
    percent = df.isnull().sum() / df.isnull().count() * 100
    table = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    table["Types"] = [str(df[col].dtype) for col in df.columns]
    return table.T


def notebook_frequent_reference(df):
    """Reproduce notebook cells 20/21, with the requested all-null behavior."""
    total = df.count()
    table = pd.DataFrame(total)
    table.columns = ["Total"]
    items, vals = [], []
    for col in df.columns:
        counts = df[col].value_counts()
        if counts.empty:
            items.append(pd.NA)
            vals.append(0)
        else:
            items.append(counts.index[0])
            vals.append(counts.iloc[0])
    table["Most frequent item"] = items
    table["Frequence"] = vals
    table["Percent from total"] = np.round(vals / total * 100, 3)
    return table.T


def notebook_unique_reference(df):
    """Reproduce notebook cells 23/24."""
    total = df.count()
    table = pd.DataFrame(total)
    table.columns = ["Total"]
    table["Uniques"] = [df[col].nunique() for col in df.columns]
    return table.T


class DataLoadingTests(unittest.TestCase):
    def test_loads_training_and_test_csvs_like_read_csv(self):
        for filename in ("train.csv", "test.csv"):
            with self.subTest(filename=filename):
                expected = pd.read_csv(ROOT / "data" / filename)
                actual = load_data(ROOT / "data" / filename)
                assert_frame_equal(actual, expected)

    def test_missing_path_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_data(ROOT / "data" / "does-not-exist.csv")


class DataInspectionTests(unittest.TestCase):
    def test_summaries_match_notebook_formulas_on_real_csvs(self):
        for filename in ("train.csv", "test.csv"):
            df = pd.read_csv(ROOT / "data" / filename)
            with self.subTest(filename=filename):
                assert_frame_equal(calculate_missing_values(df), notebook_missing_reference(df))
                assert_frame_equal(get_frequent_data(df), notebook_frequent_reference(df))
                assert_frame_equal(get_unique_values(df), notebook_unique_reference(df))

    def test_inspect_data_contains_head_info_and_default_describe(self):
        df = pd.DataFrame({"number": [1, 2, 3], "label": ["a", "b", "a"]})
        before = df.copy(deep=True)
        result = inspect_data(df)

        self.assertEqual(set(result), {"head", "info", "describe"})
        assert_frame_equal(result["head"], df.head())
        info = io.StringIO()
        df.info(buf=info)
        self.assertEqual(result["info"], info.getvalue())
        assert_frame_equal(result["describe"], df.describe())
        assert_frame_equal(df, before)

    def test_inspection_handles_zero_column_frame(self):
        empty = pd.DataFrame(index=range(2))
        result = inspect_data(empty)
        self.assertEqual(set(result), {"head", "info", "describe"})
        assert_frame_equal(result["head"], empty.head())
        assert_frame_equal(result["describe"], pd.DataFrame())

    def test_missing_values_counts_all_rows_and_reports_dtype(self):
        df = pd.DataFrame({"mixed": [1, None, 3, None], "complete": ["x", "y", "x", "z"]})
        before = df.copy(deep=True)
        result = calculate_missing_values(df)
        expected = pd.DataFrame(
            [[2, 0], [50.0, 0.0], ["float64", "object"]],
            index=["Total", "Percent", "Types"],
            columns=["mixed", "complete"],
        )
        assert_frame_equal(result, expected)
        assert_frame_equal(df, before)

    def test_frequent_and_unique_summaries_handle_ties_and_all_null(self):
        df = pd.DataFrame({"tie": ["b", "a", "b", "a"], "all_null": [None] * 4})
        before = df.copy(deep=True)
        frequent = get_frequent_data(df)
        uniques = get_unique_values(df)
        expected_freq = pd.DataFrame(
            [[4, 0], ["b", pd.NA], [2, 0], [50.0, np.nan]],
            index=["Total", "Most frequent item", "Frequence", "Percent from total"],
            columns=["tie", "all_null"],
        )
        expected_unique = pd.DataFrame(
            [[4, 0], [2, 0]], index=["Total", "Uniques"], columns=["tie", "all_null"]
        )
        assert_frame_equal(frequent, expected_freq)
        assert_frame_equal(uniques, expected_unique)
        assert_frame_equal(df, before)

    def test_unobserved_categories_are_not_reported_as_frequent(self):
        df = pd.DataFrame({"category": pd.Categorical([None, None], categories=["a", "b"])})
        result = get_frequent_data(df)
        self.assertTrue(pd.isna(result.loc["Most frequent item", "category"]))
        self.assertEqual(result.loc["Frequence", "category"], 0)
        self.assertTrue(pd.isna(result.loc["Percent from total", "category"]))

    def test_zero_row_columns_have_safe_empty_summary_values(self):
        df = pd.DataFrame({"number": pd.Series(dtype="int64"), "text": pd.Series(dtype="object")})
        missing = calculate_missing_values(df)
        frequent = get_frequent_data(df)
        uniques = get_unique_values(df)

        self.assertEqual(missing.loc["Total"].tolist(), [0, 0])
        self.assertTrue(missing.loc["Percent"].isna().all())
        self.assertEqual(missing.loc["Types"].tolist(), ["int64", "object"])
        self.assertEqual(frequent.loc["Total"].tolist(), [0, 0])
        self.assertTrue(frequent.loc["Most frequent item"].isna().all())
        self.assertEqual(frequent.loc["Frequence"].tolist(), [0, 0])
        self.assertTrue(frequent.loc["Percent from total"].isna().all())
        self.assertEqual(uniques.loc["Total"].tolist(), [0, 0])
        self.assertEqual(uniques.loc["Uniques"].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
