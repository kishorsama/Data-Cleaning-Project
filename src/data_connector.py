import os
import io
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, inspect

class DataConnector:
    """
    Universal ingestion and export engine for flat files and SQL databases.
    """

    @staticmethod
    def read_file(file_obj_or_path, file_name: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Reads CSV, TSV, Excel, and Parquet files into a DataFrame.
        """
        ext = os.path.splitext(file_name)[1].lower()

        if ext in ('.csv', '.txt'):
            try:
                return pd.read_csv(file_obj_or_path)
            except UnicodeDecodeError:
                if hasattr(file_obj_or_path, 'seek'):
                    file_obj_or_path.seek(0)
                return pd.read_csv(file_obj_or_path, encoding='latin1')
        elif ext == '.tsv':
            return pd.read_csv(file_obj_or_path, sep='\t')
        elif ext in ('.xlsx', '.xls'):
            return pd.read_excel(file_obj_or_path, sheet_name=sheet_name or 0)
        elif ext == '.parquet':
            return pd.read_parquet(file_obj_or_path)
        else:
            # Fallback attempt as csv
            return pd.read_csv(file_obj_or_path)

    @staticmethod
    def get_database_engine(connection_uri_or_path: str):
        """
        Creates a SQLAlchemy engine from a connection URI or SQLite file path.
        """
        uri = connection_uri_or_path.strip()
        if not (uri.startswith("sqlite:///") or uri.startswith("postgresql://") or uri.startswith("mysql")):
            if os.path.exists(uri) and uri.endswith(('.db', '.sqlite', '.sqlite3')):
                uri = f"sqlite:///{os.path.abspath(uri)}"
        return create_engine(uri)

    @staticmethod
    def list_database_tables(connection_uri_or_path: str) -> List[str]:
        """
        Lists table names from a database connection.
        """
        engine = DataConnector.get_database_engine(connection_uri_or_path)
        inspector = inspect(engine)
        return inspector.get_table_names()

    @staticmethod
    def load_database_table(connection_uri_or_path: str, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Loads a table from a database into a DataFrame.
        """
        engine = DataConnector.get_database_engine(connection_uri_or_path)
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {int(limit)}"
        return pd.read_sql(query, con=engine)

    @staticmethod
    def write_database_table(df: pd.DataFrame, connection_uri_or_path: str, table_name: str, if_exists: str = "replace") -> bool:
        """
        Exports a cleaned DataFrame into a database table.
        """
        engine = DataConnector.get_database_engine(connection_uri_or_path)
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        return True
