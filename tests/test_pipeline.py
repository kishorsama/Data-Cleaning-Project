import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure src in path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_connector import DataConnector
from profiler import get_dataset_profile, compare_profiles
from null_handling import handle_nulls
from deduplication import deduplicate, remove_duplicates
from outlier_detection import handle_outliers
from agent_graph import run_autonomous_clean

class TestProductionPipeline(unittest.TestCase):

    def setUp(self):
        # Synthetic dirty dataset
        np.random.seed(42)
        n = 100
        data = {
            "id": list(range(1, n + 1)),
            "age": [20, 25, 30, np.nan, 22, 120, -5, 28, np.nan, 35] * 10,
            "salary": [50000, 60000, np.nan, 80000, 55000, 1000000, 62000, np.nan, 75000, 90000] * 10,
            "category": ["A", "B", np.nan, "A", "C", "B", "A", "C", np.nan, "B"] * 10,
            "useless_empty_col": [np.nan] * 95 + [1.0, 2.0, 3.0, 4.0, 5.0]  # 95% nulls
        }
        self.dirty_df = pd.DataFrame(data)
        # Add intentional duplicates
        self.dirty_df = pd.concat([self.dirty_df, self.dirty_df.iloc[:10]], ignore_index=True)

    def test_profiler_and_scoring(self):
        profile = get_dataset_profile(self.dirty_df)
        self.assertEqual(profile["total_rows"], 110)
        self.assertEqual(profile["duplicate_rows"], 10)
        self.assertGreater(profile["total_null_cells"], 0)
        self.assertGreater(profile["quality_score"], 0)
        self.assertLessEqual(profile["quality_score"], 100)

    def test_null_handling_auto(self):
        cleaned = handle_nulls(self.dirty_df, strategy="auto")
        # useless_empty_col had 95% nulls so it should be dropped
        self.assertNotIn("useless_empty_col", cleaned.columns)
        # Remaining columns should have zero nulls
        self.assertEqual(cleaned["age"].isnull().sum(), 0)
        self.assertEqual(cleaned["salary"].isnull().sum(), 0)
        self.assertEqual(cleaned["category"].isnull().sum(), 0)

    def test_deduplication(self):
        cleaned = deduplicate(self.dirty_df)
        self.assertEqual(len(cleaned), 100)
        self.assertEqual(cleaned.duplicated().sum(), 0)

    def test_outlier_detection(self):
        # Check IQR clipping on salary (which has 1,000,000)
        cleaned = handle_outliers(self.dirty_df, columns=["salary"])
        self.assertLess(cleaned["salary"].max(), 1000000)

    def test_sqlite_connector(self):
        test_db = os.path.join(current_dir, "test_temp.db")
        db_uri = f"sqlite:///{os.path.abspath(test_db)}"
        try:
            # Write DataFrame to database
            DataConnector.write_database_table(self.dirty_df.head(20), db_uri, "test_table")
            # List tables
            tables = DataConnector.list_database_tables(db_uri)
            self.assertIn("test_table", tables)
            # Read back table
            read_df = DataConnector.load_database_table(db_uri, "test_table")
            self.assertEqual(len(read_df), 20)
        finally:
            if os.path.exists(test_db):
                try:
                    os.remove(test_db)
                except Exception:
                    pass

    def test_autonomous_agent_pipeline(self):
        result = run_autonomous_clean(self.dirty_df, autonomous_mode=True)
        clean_df = result["df"]
        report = result["quality_report"]

        # Validate pipeline cleaned everything
        self.assertEqual(clean_df.duplicated().sum(), 0)
        self.assertEqual(clean_df.isnull().sum().sum(), 0)
        self.assertGreaterEqual(report["quality_score_after"], report["quality_score_before"])
        self.assertGreater(report["duplicates_eliminated"], 0)
        self.assertGreater(report["nulls_eliminated"], 0)

if __name__ == "__main__":
    unittest.main()
