import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def compute_health_score(
    total_cells: int, 
    null_cells: int, 
    total_rows: int, 
    duplicate_rows: int, 
    total_numeric_cells: int, 
    outlier_cells: int
) -> float:
    """
    Computes a realistic 0 to 100 data health score based on completeness,
    uniqueness, and numeric distributions.
    """
    if total_cells == 0 or total_rows == 0:
        return 0.0

    score = 100.0

    # Penalize missing values (up to 40 points)
    null_rate = null_cells / total_cells
    score -= min(40.0, null_rate * 100.0 * 0.8)

    # Penalize duplicate rows (up to 25 points)
    dup_rate = duplicate_rows / total_rows
    score -= min(25.0, dup_rate * 100.0 * 0.8)

    # Penalize extreme outliers (up to 20 points)
    if total_numeric_cells > 0:
        outlier_rate = outlier_cells / total_numeric_cells
        score -= min(20.0, outlier_rate * 100.0 * 0.7)

    return max(5.0, round(score, 1))


def detect_semantic_type(series: pd.Series, col_name: str) -> str:
    """
    Infers the high-level semantic meaning of a column to guide cleaning decisions.
    """
    clean_series = series.dropna()
    n_unique = clean_series.nunique()
    n_total = len(clean_series)

    if n_total == 0:
        return "empty"

    # Datetime detection
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    name_lower = col_name.lower()
    if any(k in name_lower for k in ["date", "time", "timestamp", "year", "dob"]):
        if series.dtype == 'object':
            try:
                sample = clean_series.iloc[:10].astype(str)
                pd.to_datetime(sample, errors='raise')
                return "datetime"
            except Exception:
                pass

    if pd.api.types.is_bool_dtype(series) or (n_unique == 2 and n_total > 5):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        if n_unique < 15 and pd.api.types.is_integer_dtype(series):
            return "numeric_discrete"
        return "numeric_continuous"

    if series.dtype == 'object' or str(series.dtype) == 'category':
        uniqueness_ratio = n_unique / n_total
        # High cardinality ID column
        if uniqueness_ratio > 0.85 and any(k in name_lower for k in ["id", "uuid", "key", "code"]):
            return "id_high_cardinality"
        elif n_unique <= 30:
            return "categorical"
        return "text"

    return "general"


def get_dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a statistical and structural profile of the DataFrame
    to send to the planner without transferring raw row data.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_mb = round(memory_bytes / (1024 * 1024), 2)

    total_cells = total_rows * total_cols
    total_null_cells = int(df.isnull().sum().sum())
    total_numeric_cells = 0
    total_outliers = 0

    columns_meta = {}

    for col in df.columns:
        series = df[col]
        is_num = pd.api.types.is_numeric_dtype(series)
        null_count = int(series.isnull().sum())
        non_nulls = series.dropna()
        n_unique = int(series.nunique())
        semantic = detect_semantic_type(series, col)

        col_meta = {
            "dtype": str(series.dtype),
            "semantic_type": semantic,
            "null_count": null_count,
            "null_pct": round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0,
            "unique_count": n_unique,
            "sample_values": [str(x) for x in non_nulls.unique()[:3].tolist()]
        }

        if is_num and len(non_nulls) > 0:
            total_numeric_cells += len(non_nulls)
            q1 = float(non_nulls.quantile(0.25))
            q3 = float(non_nulls.quantile(0.75))
            iqr = q3 - q1
            skew = round(float(non_nulls.skew()), 2) if len(non_nulls) > 2 else 0.0

            outliers_count = 0
            if iqr > 0:
                outliers_count = int(((non_nulls < (q1 - 1.5 * iqr)) | (non_nulls > (q3 + 1.5 * iqr))).sum())
            total_outliers += outliers_count

            col_meta.update({
                "skewness": skew,
                "mean": round(float(non_nulls.mean()), 2),
                "median": round(float(non_nulls.median()), 2),
                "std": round(float(non_nulls.std()), 2) if len(non_nulls) > 1 else 0.0,
                "min": round(float(non_nulls.min()), 2),
                "max": round(float(non_nulls.max()), 2),
                "outliers_count": outliers_count
            })
        else:
            col_meta["skewness"] = None
            col_meta["outliers_count"] = 0

        columns_meta[col] = col_meta

    health_score = compute_health_score(
        total_cells=total_cells,
        null_cells=total_null_cells,
        total_rows=total_rows,
        duplicate_rows=duplicate_rows,
        total_numeric_cells=total_numeric_cells,
        outlier_cells=total_outliers
    )

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "total_null_cells": total_null_cells,
        "null_cell_pct": round((total_null_cells / total_cells * 100), 2) if total_cells > 0 else 0.0,
        "total_outliers": total_outliers,
        "memory_mb": memory_mb,
        "quality_score": health_score,
        "columns": columns_meta
    }


def compare_profiles(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates differences between the pre-cleaning and post-cleaning profiles.
    """
    rows_delta = after.get("total_rows", 0) - before.get("total_rows", 0)
    cols_delta = after.get("total_columns", 0) - before.get("total_columns", 0)
    nulls_eliminated = before.get("total_null_cells", 0) - after.get("total_null_cells", 0)
    duplicates_eliminated = before.get("duplicate_rows", 0) - after.get("duplicate_rows", 0)
    outliers_handled = before.get("total_outliers", 0) - after.get("total_outliers", 0)

    mem_before = before.get("memory_mb", 0.0)
    mem_after = after.get("memory_mb", 0.0)
    mem_saved_mb = round(mem_before - mem_after, 2)
    mem_saved_pct = round((mem_saved_mb / mem_before * 100), 1) if mem_before > 0 else 0.0

    score_before = before.get("quality_score", 0.0)
    score_after = after.get("quality_score", 0.0)
    score_improvement = round(score_after - score_before, 1)

    return {
        "rows_delta": rows_delta,
        "cols_delta": cols_delta,
        "nulls_eliminated": max(0, nulls_eliminated),
        "duplicates_eliminated": max(0, duplicates_eliminated),
        "outliers_handled": max(0, outliers_handled),
        "memory_saved_mb": mem_saved_mb,
        "memory_saved_pct": max(0.0, mem_saved_pct),
        "quality_score_before": score_before,
        "quality_score_after": score_after,
        "score_improvement": score_improvement
    }
