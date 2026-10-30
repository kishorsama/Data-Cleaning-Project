import os
import sys
import io
import uuid
import streamlit as st
import pandas as pd

# Portable local import setup
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data_connector import DataConnector
from profiler import get_dataset_profile
from agent_graph import run_autonomous_clean, cleaning_graph, set_cached_df, get_cached_df
from langgraph.types import Command

# Data cleaning modules
from null_handling import handle_nulls
from outlier_detection import handle_outliers
from deduplication import deduplicate
from typecasting import typecasting

# Data transformation modules
from label_encoding import label_encoding
from normalization import minmax, zscore
from discretization import equal_width, equal_frequency, kbins

# Data reduction modules
from dimensionality import pca
from numerosity import random_sampling, stratified_sampling
from compression import lossy, lossless


def render_export(df: pd.DataFrame):
    """Renders export buttons for CSV, Excel, and SQL export."""
    st.markdown("---")
    st.subheader("Export Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Cleaned')
            st.download_button(
                label="Download as Excel",
                data=buf.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )
        except Exception:
            st.download_button(
                label="Download as Excel (CSV format)",
                data=csv_data,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    with col3:
        with st.popover("Save to Database"):
            db_uri = st.text_input("Database URI / Path", placeholder="sqlite:///cleaned_data.db")
            table = st.text_input("Table Name", value="cleaned_table")
            if st.button("Save Table", use_container_width=True):
                if db_uri and table:
                    try:
                        DataConnector.write_database_table(df, db_uri, table)
                        st.success(f"Saved {len(df):,} rows to table '{table}'")
                    except Exception as err:
                        st.error(f"Save failed: {err}")
                else:
                    st.warning("Provide both a URI and a table name.")


def main():
    st.set_page_config(
        page_title="Autonomous Data Cleaning & Preprocessing",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Session state initialization
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'original_df' not in st.session_state:
        st.session_state['original_df'] = None
    if 'agent_result' not in st.session_state:
        st.session_state['agent_result'] = None
    if 'thread_id' not in st.session_state:
        st.session_state['thread_id'] = None
    if 'awaiting_approval' not in st.session_state:
        st.session_state['awaiting_approval'] = False

    # Sidebar Navigation
    st.sidebar.title("Data Pipeline")
    pages = [
        "1. Ingestion", 
        "2. Profiler & Health", 
        "3. Manual Cleaning", 
        "4. Transformation", 
        "5. Reduction",
        "6. Autonomous Agent"
    ]
    page = st.sidebar.radio("Navigate:", pages)

    if st.session_state['original_df'] is not None:
        st.sidebar.markdown("---")
        st.sidebar.caption(
            f"Active: **{len(st.session_state['df']):,} rows × {len(st.session_state['df'].columns)} cols**"
        )
        if st.sidebar.button("Reset to Original Data", use_container_width=True):
            st.session_state['df'] = st.session_state['original_df'].copy()
            st.session_state['agent_result'] = None
            st.session_state['awaiting_approval'] = False
            st.sidebar.success("Reset data to original uploaded state.")
            st.rerun()

    # -------------------------------------------------------------
    # PAGE 1: INGESTION
    # -------------------------------------------------------------
    if page == pages[0]:
        st.title("Dataset Ingestion")
        st.write("Upload a file (CSV, Excel, Parquet) or connect to an external SQL database.")

        tab_upload, tab_db, tab_sample = st.tabs(["Upload File", "Connect Database", "Sample Datasets"])

        with tab_upload:
            uploaded = st.file_uploader("Choose a dataset file", type=["csv", "xlsx", "xls", "parquet", "tsv"])
            if uploaded is not None:
                try:
                    df = DataConnector.read_file(uploaded, uploaded.name)
                    st.session_state['original_df'] = df.copy()
                    st.session_state['df'] = df.copy()
                    st.session_state['agent_result'] = None
                    st.success(f"Loaded '{uploaded.name}' ({len(df):,} rows × {len(df.columns)} columns)")
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        with tab_db:
            db_uri = st.text_input("Connection URI", placeholder="sqlite:///path/to/data.db or postgresql://user:pass@host/db")
            col_btn, col_lim = st.columns([3, 1])
            with col_btn:
                fetch = st.button("List Tables", use_container_width=True)
            with col_lim:
                limit = st.number_input("Max Rows (0 = All)", min_value=0, value=10000, step=1000)

            if fetch and db_uri:
                try:
                    tables = DataConnector.list_database_tables(db_uri)
                    if tables:
                        st.session_state['db_tables'] = tables
                        st.session_state['active_uri'] = db_uri
                        st.success(f"Found {len(tables)} tables: {', '.join(tables)}")
                    else:
                        st.info("Connected, but no tables found in database.")
                except Exception as ex:
                    st.error(f"Connection failed: {ex}")

            if 'db_tables' in st.session_state and st.session_state.get('active_uri') == db_uri:
                table_choice = st.selectbox("Select Table", st.session_state['db_tables'])
                if st.button("Load Table Data", type="primary"):
                    try:
                        row_lim = limit if limit > 0 else None
                        df_db = DataConnector.load_database_table(db_uri, table_choice, limit=row_lim)
                        st.session_state['original_df'] = df_db.copy()
                        st.session_state['df'] = df_db.copy()
                        st.session_state['agent_result'] = None
                        st.success(f"Loaded '{table_choice}' ({len(df_db):,} rows × {len(df_db.columns)} columns)")
                    except Exception as err:
                        st.error(f"Error reading table: {err}")

        with tab_sample:
            dataset_dir = os.path.join(os.path.dirname(base_dir), "Dataset")
            samples = {}
            if os.path.exists(dataset_dir):
                for f in os.listdir(dataset_dir):
                    if f.endswith('.csv'):
                        samples[f] = os.path.join(dataset_dir, f)
            choice = st.selectbox("Choose Sample Dataset", list(samples.keys()) if samples else ["None"])
            if st.button("Load Sample") and samples:
                df_sample = pd.read_csv(samples[choice])
                st.session_state['original_df'] = df_sample.copy()
                st.session_state['df'] = df_sample.copy()
                st.session_state['agent_result'] = None
                st.success(f"Loaded '{choice}' ({len(df_sample):,} rows × {len(df_sample.columns)} columns)")

        if st.session_state['df'] is not None:
            st.markdown("---")
            st.subheader("Data Preview")
            st.dataframe(st.session_state['df'].head(10), use_container_width=True)
            render_export(st.session_state['df'])

    # -------------------------------------------------------------
    # PAGE 2: PROFILER & HEALTH
    # -------------------------------------------------------------
    elif page == pages[1]:
        st.title("Data Profiler & Health Diagnosis")
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            profile = get_dataset_profile(df)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Health Score", f"{profile['quality_score']}/100")
            c2.metric("Total Rows", f"{profile['total_rows']:,}")
            c3.metric("Total Columns", profile['total_columns'])
            c4.metric("Duplicates", f"{profile['duplicate_rows']} ({profile['duplicate_pct']}%)")
            c5.metric("Missing Cells", f"{profile['total_null_cells']:,} ({profile['null_cell_pct']}%)")

            st.markdown("### Column Analysis")
            col_rows = []
            for col, meta in profile["columns"].items():
                col_rows.append({
                    "Column": col,
                    "Type": meta["dtype"],
                    "Inferred Role": meta["semantic_type"],
                    "Missing": meta["null_count"],
                    "Missing %": meta["null_pct"],
                    "Distinct": meta["unique_count"],
                    "Skewness": meta["skewness"],
                    "Outliers (IQR)": meta["outliers_count"],
                    "Sample Values": ", ".join(meta["sample_values"][:3])
                })
            st.dataframe(pd.DataFrame(col_rows), use_container_width=True)

            st.markdown("### Summary Statistics")
            st.dataframe(df.describe(include='all').fillna(""), use_container_width=True)
            render_export(df)
        else:
            st.warning("Upload or connect a dataset in Step 1 to view profiling.")

    # -------------------------------------------------------------
    # PAGE 3: MANUAL CLEANING
    # -------------------------------------------------------------
    elif page == pages[2]:
        st.title("Manual Data Cleaning")
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            step = st.selectbox("Operation:", ["Deduplicate", "Handle Nulls", "Clip Outliers (IQR)", "Type Casting"])

            if step == "Deduplicate":
                dups = df.duplicated().sum()
                st.write(f"Identified duplicate rows: **{dups}**")
                if st.button("Drop Duplicates", type="primary"):
                    st.session_state['df'] = deduplicate(df)
                    st.success(f"Removed {dups} duplicate rows.")
                    st.rerun()

            elif step == "Handle Nulls":
                null_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
                if not null_cols:
                    st.success("No missing values found.")
                else:
                    selected = st.multiselect("Columns", null_cols, default=null_cols)
                    strat = st.selectbox("Imputation Strategy:", ["auto", "median", "mean", "mode", "drop", "ffill", "bfill", "interpolate"])
                    if st.button("Apply Imputation", type="primary"):
                        st.session_state['df'] = handle_nulls(df, strategy=strat, columns=selected)
                        st.success(f"Imputed missing values using strategy '{strat}'.")
                        st.rerun()

            elif step == "Clip Outliers (IQR)":
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                selected = st.multiselect("Target Numeric Columns", num_cols, default=num_cols)
                factor = st.slider("Multiplier (default 1.5):", 1.0, 3.0, 1.5, 0.1)
                if st.button("Clip Outliers", type="primary"):
                    st.session_state['df'] = handle_outliers(df, columns=selected, threshold=factor)
                    st.success(f"Clipped outliers across {len(selected)} column(s).")
                    st.rerun()

            elif step == "Type Casting":
                col = st.selectbox("Column", df.columns)
                target_type = st.selectbox("Cast To", ["datetime64[ns]", "str", "int64", "float64", "category"])
                if st.button("Cast Column", type="primary"):
                    try:
                        st.session_state['df'] = typecasting(df, [col], target_type)
                        st.success(f"Converted '{col}' to '{target_type}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Typecast failed: {e}")

            st.divider()
            st.dataframe(st.session_state['df'].head(), use_container_width=True)
            render_export(st.session_state['df'])
        else:
            st.warning("Please load data in Step 1 first.")

    # -------------------------------------------------------------
    # PAGE 4: TRANSFORMATION
    # -------------------------------------------------------------
    elif page == pages[3]:
        st.title("Data Transformation")
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            trans = st.selectbox("Transformation:", ["Label Encoding", "Min-Max Normalization", "Z-Score Standardization", "Discretization"])

            if trans == "Label Encoding":
                cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
                selected = st.multiselect("Categorical Columns", cats, default=cats)
                if st.button("Apply Label Encoding", type="primary") and selected:
                    clean_df, mapping = label_encoding(df, selected)
                    st.session_state['df'] = clean_df
                    st.success("Label encoding applied.")
                    with st.expander("Show Mappings"):
                        st.json(mapping)
                    st.rerun()

            elif trans == "Min-Max Normalization":
                nums = df.select_dtypes(include=['number']).columns.tolist()
                selected = st.multiselect("Numeric Columns", nums, default=nums)
                if st.button("Normalize (0 to 1)", type="primary") and selected:
                    st.session_state['df'] = minmax(df, selected)
                    st.success("Min-Max normalization applied.")
                    st.rerun()

            elif trans == "Z-Score Standardization":
                nums = df.select_dtypes(include=['number']).columns.tolist()
                selected = st.multiselect("Numeric Columns", nums, default=nums)
                if st.button("Standardize", type="primary") and selected:
                    st.session_state['df'] = zscore(df, selected)
                    st.success("Z-score standardization applied.")
                    st.rerun()

            elif trans == "Discretization":
                nums = df.select_dtypes(include=['number']).columns.tolist()
                target_col = st.selectbox("Column to Bin", nums)
                method = st.radio("Binning Strategy:", ["Equal Width", "Equal Frequency", "K-Means"])
                bins = st.slider("Bins:", 2, 10, 4)
                if st.button("Discretize", type="primary") and target_col:
                    if method == "Equal Width":
                        st.session_state['df'] = equal_width(df, target_col, bin_number=bins)
                    elif method == "Equal Frequency":
                        st.session_state['df'] = equal_frequency(df, target_col, quartile_number=bins)
                    else:
                        st.session_state['df'] = kbins(df, [target_col], bin_number=bins)
                    st.success(f"Column '{target_col}' binned into {bins} buckets.")
                    st.rerun()

            st.divider()
            st.dataframe(st.session_state['df'].head(), use_container_width=True)
            render_export(st.session_state['df'])
        else:
            st.warning("Please load data in Step 1 first.")

    # -------------------------------------------------------------
    # PAGE 5: REDUCTION
    # -------------------------------------------------------------
    elif page == pages[4]:
        st.title("Data Reduction")
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            red = st.selectbox("Technique:", ["Lossless Memory Compression", "Lossy Downcasting", "PCA", "Sampling"])

            if red == "Lossless Memory Compression":
                if st.button("Compress Memory", type="primary"):
                    st.session_state['df'] = lossless(df)
                    st.success("Compressed memory without precision loss.")
                    st.rerun()

            elif red == "Lossy Downcasting":
                if st.button("Downcast Types", type="primary"):
                    st.session_state['df'] = lossy(df)
                    st.success("Downcasted numerical types.")
                    st.rerun()

            elif red == "PCA":
                nums = df.select_dtypes(include=['number']).columns.tolist()
                selected = st.multiselect("Features", nums, default=nums)
                n = st.slider("Components:", 1, min(len(selected), 10) if selected else 2, 2)
                if st.button("Compute PCA", type="primary") and selected:
                    st.session_state['df'] = pca(df, selected, n=n)
                    st.success(f"Reduced to {n} PCA components.")
                    st.rerun()

            elif red == "Sampling":
                mode = st.radio("Mode:", ["Random", "Stratified"])
                frac = st.slider("Fraction:", 0.05, 0.95, 0.5, 0.05)
                if mode == "Random":
                    if st.button("Sample Data", type="primary"):
                        st.session_state['df'] = random_sampling(df, fract=frac)
                        st.success(f"Sampled {frac*100:.0f}% of rows.")
                        st.rerun()
                else:
                    cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    strat_col = st.selectbox("Stratify By:", cats)
                    if st.button("Sample Data", type="primary") and strat_col:
                        st.session_state['df'] = stratified_sampling(df, strat_col, fract=frac)
                        st.success(f"Stratified sample on '{strat_col}' applied.")
                        st.rerun()

            st.divider()
            st.dataframe(st.session_state['df'].head(), use_container_width=True)
            render_export(st.session_state['df'])
        else:
            st.warning("Please load data in Step 1 first.")

    # -------------------------------------------------------------
    # PAGE 6: AUTONOMOUS AGENT
    # -------------------------------------------------------------
    elif page == pages[5]:
        st.title("Autonomous Cleaning Agent")
        st.write("LangGraph-driven agent that profiles data, formulates an optimal plan, and executes it.")

        if st.session_state['df'] is not None:
            df_active = st.session_state['df']

            col_mode, col_info = st.columns([2, 3])
            with col_mode:
                mode = st.radio("Operating Mode:", ["Auto-Pilot (Fully Autonomous)", "Co-Pilot (Review Plan Before Execution)"])
            with col_info:
                st.caption(
                    "**Auto-Pilot**: Detects anomalies and cleans immediately.\n"
                    "**Co-Pilot**: Pauses with proposed plan for your review and edits."
                )

            is_autopilot = "Auto-Pilot" in mode

            if not st.session_state['awaiting_approval']:
                if st.button("Run Cleaning Agent", type="primary", use_container_width=True):
                    with st.spinner("Profiling dataset and assembling cleaning plan..."):
                        if is_autopilot:
                            result = run_autonomous_clean(df_active, autonomous_mode=True)
                            st.session_state['df'] = result['df']
                            st.session_state['agent_result'] = result
                            st.success("Cleaning completed successfully.")
                            st.rerun()
                        else:
                            thread_id = str(uuid.uuid4())
                            st.session_state['thread_id'] = thread_id
                            set_cached_df(thread_id, df_active.copy())
                            cfg = {"configurable": {"thread_id": thread_id}}
                            initial_state = {
                                "dataset_id": thread_id,
                                "autonomous_mode": False,
                                "attempts": 0,
                                "logs": [],
                                "errors": []
                            }
                            cleaning_graph.invoke(initial_state, config=cfg)
                            st.session_state['awaiting_approval'] = True
                            st.rerun()

            # Review checkpoint in Co-Pilot mode
            if st.session_state['awaiting_approval'] and st.session_state['thread_id']:
                cfg = {"configurable": {"thread_id": st.session_state['thread_id']}}
                state = cleaning_graph.get_state(cfg)

                if state.values and "plan" in state.values:
                    st.subheader("Proposed Plan")
                    for i, step in enumerate(state.values["plan"], 1):
                        with st.expander(f"Step {i}: {step['operation']} on {step.get('columns') or 'All columns'}", expanded=True):
                            st.write(f"**Rationale:** {step['reason']}")
                            if step.get('strategy'):
                                st.caption(f"Strategy: `{step['strategy']}` | Threshold: `{step.get('threshold', 1.5)}`")

                    c_app, c_rep, c_can = st.columns(3)
                    with c_app:
                        if st.button("Approve & Execute", type="primary", use_container_width=True):
                            with st.spinner("Executing approved steps..."):
                                final = cleaning_graph.invoke(
                                    Command(resume={"action": "approve", "plan": state.values["plan"]}),
                                    config=cfg
                                )
                                final_df = get_cached_df(st.session_state['thread_id'])
                                final["df"] = final_df
                                st.session_state['df'] = final_df
                                st.session_state['agent_result'] = final
                                st.session_state['awaiting_approval'] = False
                                st.success("Approved plan executed successfully.")
                                st.rerun()
                    with c_rep:
                        if st.button("Re-Plan", use_container_width=True):
                            with st.spinner("Re-evaluating profile..."):
                                cleaning_graph.invoke(Command(resume={"action": "replan"}), config=cfg)
                                st.rerun()
                    with c_can:
                        if st.button("Cancel", use_container_width=True):
                            st.session_state['awaiting_approval'] = False
                            st.rerun()

            # Scorecard and comparison
            if st.session_state.get('agent_result'):
                res = st.session_state['agent_result']
                report = res.get('quality_report', {})

                st.markdown("---")
                st.subheader("Quality Improvement Scorecard")

                m1, m2, m3, m4, m5 = st.columns(5)
                before = report.get('quality_score_before', 0.0)
                after = report.get('quality_score_after', 0.0)
                diff = report.get('score_improvement', 0.0)

                m1.metric("Health Score", f"{after}/100", f"+{diff} pts" if diff > 0 else f"{diff} pts")
                m2.metric("Nulls Removed", f"{report.get('nulls_eliminated', 0):,}")
                m3.metric("Duplicates Dropped", f"{report.get('duplicates_eliminated', 0):,}")
                m4.metric("Outliers Clipped", f"{report.get('outliers_handled', 0):,}")
                m5.metric("RAM Saved", f"{report.get('memory_saved_pct', 0.0)}%", f"-{report.get('memory_saved_mb', 0.0)} MB")

                st.markdown("### Execution Audit Log")
                for log in res.get("logs", []):
                    st.text(log)

                st.markdown("### Dataset Comparison")
                tab_after, tab_before = st.tabs(["Cleaned Dataset", "Original Dataset"])
                with tab_after:
                    st.dataframe(st.session_state['df'].head(15), use_container_width=True)
                with tab_before:
                    st.dataframe(st.session_state['original_df'].head(15), use_container_width=True)

                render_export(st.session_state['df'])
        else:
            st.warning("Please upload or connect a dataset in Step 1 first.")


if __name__ == '__main__':
    main()
