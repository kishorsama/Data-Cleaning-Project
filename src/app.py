import streamlit as st
import pandas as pd
import io
import sys
sys.path.append('/workspaces/Data-Cleaning-Project/src')

#data cleaning
from null_handling import drop, forward_fill, backward_fill, mean_filling, median_filling, mode_filling, interpolate_fill 
from outlier_detection import handle_outliers
from deduplication import remove_duplicates
from typecasting import typecasting

#data transformation
from label_encoding import label_encoding
from normalization import minmax, zscore
from scaling import min_max_scaling, standard_scaling
from discretization import equal_width, equal_frequency, kbins

#data reduction
from dimensionality import pca, tsne
from numerosity import random_sampling, stratified_sampling, cluster_reduction
from compression import lossy, lossless



def export_section(df):
    """Provides UI for exporting the dataframe to CSV and Excel."""
    st.markdown("---")
    st.subheader("💾 Export Data")
    st.write("You can download the dataset as CSV or Excel at any point.")
    
    col1, col2 = st.columns(2)
    with col1:
        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
    with col2:
        # Excel Export
        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            st.download_button(
                label="Download as Excel",
                data=buffer.getvalue(),
                file_name="processed_data.xlsx",
                mime="application/vnd.ms-excel"
            )
        except Exception as e:
            # Fallback
            st.warning("Excel export requires 'xlsxwriter'. Please save as CSV or install it.")

def main():
    st.set_page_config(page_title="Data Processing Suite", layout="wide")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'original_df' not in st.session_state:
        st.session_state['original_df'] = None

    # Sidebar Navigation
    st.sidebar.title("Navigation Steps")
    pages = [
        "1. Information & Upload", 
        "2. Data Exploration", 
        "3. Data Cleaning", 
        "4. Data Transformation", 
        "5. Data Reduction"
    ]
    choice = st.sidebar.radio("Go to step:", pages)
    
    if st.session_state['original_df'] is not None:
        if st.sidebar.button("Reset to Original Data"):
            st.session_state['df'] = st.session_state['original_df'].copy()
            st.sidebar.success("Data reset to original uploaded state.")

    # -------------------------------------------------------------
    # PAGE 1: INFORMATION & UPLOAD
    # -------------------------------------------------------------
    if choice == pages[0]:
        st.title("📊 Data Cleaning & Processing Tool")
        st.write("""
        **Welcome to the Automated Data Processing Suite!**
        
        This application allows you to perform basic and advanced data preprocessing steps interactively without writing a single line of code.
        
        **Workflow:**
        1. **Upload** your file (.csv).
        2. **Preview and Explore** details like datatypes, missing values, and stat summaries.
        3. **Clean** the dataset by handling nulls, outliers, deduplication, and typecasting.
        4. **Transform** variables via scaling or encodings.
        5. **Reduce** the dataset size via dimensionality, numerosity, and compression.
        
        *Note: You can skip any unneeded step and export the data at any point.*
        """)
        
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                st.session_state['original_df'] = uploaded_data.copy()
                st.session_state['df'] = uploaded_data.copy()
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        if st.session_state['df'] is not None:
            st.subheader("Quick Look")
            st.dataframe(st.session_state['df'].head())
            export_section(st.session_state['df'])

    # -------------------------------------------------------------
    # PAGE 2: DATA EXPLORATION
    # -------------------------------------------------------------
    elif choice == pages[1]:
        st.title("🔍 Data Exploration")
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            
            st.subheader("Dataset Preview (First 5 Rows)")
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            col1.metric("Total Rows", df.shape[0])
            col2.metric("Total Columns", df.shape[1])
            
            st.markdown("### Columns Breakdown")
            st.write("Displays data types and missing value counts for each column.")
            info_df = pd.DataFrame({
                'Data Type': df.dtypes.astype(str),
                'Missing Values': df.isnull().sum(),
                'Missing Percentage (%)': round((df.isnull().sum() / len(df)) * 100, 2)
            })
            st.dataframe(info_df, use_container_width=True)
            
            st.markdown("### Statistical Summary")
            st.write("Summary statistics for numerical and categorical columns.")
            st.dataframe(df.describe(include='all').fillna(""), use_container_width=True)
            
            export_section(df)
        else:
            st.warning("Please upload a file in **Step 1** to begin exploration.")

    # -------------------------------------------------------------
    # PAGE 3: DATA CLEANING
    # -------------------------------------------------------------
    elif choice == pages[2]:
        st.title("🧹 Data Cleaning")
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            
            st.write("Apply basic cleaning steps to your data. Below are some common operations you can check and apply.")
            
            # Deduplication
            st.subheader("1. Deduplication")
            duplicate_count = df.duplicated().sum()
            st.write(f"Duplicated rows found: **{duplicate_count}**")
            if st.button("Remove Duplicate Rows"):
                try:
                    new_df = remove_duplicates(df)
                    st.session_state['df'] = new_df
                    st.success(f"Removed {duplicate_count} duplicate rows.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Deduplication failed: {e}")

            st.divider()

            # Null value handling
            st.subheader("2. Null Value Handling")
            missing_total = int(df.isnull().sum().sum())
            st.write(f"Total missing values found: **{missing_total}**")
            if missing_total == 0:
                st.write("✅ Great! No missing values detected in the dataset.")
            else:
                method = st.radio(
                    "Choose a strategy to handle null values:",
                    [
                        "Drop Rows with Nulls",
                        "Drop Columns with Nulls",
                        "Forward Fill",
                        "Backward Fill",
                        "Fill with Mean",
                        "Fill with Median",
                        "Fill with Mode (Numeric)",
                        "Fill with Mode (Categorical)",
                        "Interpolate Numeric Values",
                    ],
                )

                if st.button("Apply Null Handling"):
                    try:
                        if method == "Drop Rows with Nulls":
                            new_df = drop(df, 0)
                        elif method == "Drop Columns with Nulls":
                            new_df = drop(df, 1)
                        elif method == "Forward Fill":
                            new_df = forward_fill(df, 0)
                        elif method == "Backward Fill":
                            new_df = backward_fill(df, 0)
                        elif method == "Fill with Mean":
                            new_df = mean_filling(df)
                        elif method == "Fill with Median":
                            new_df = median_filling(df)
                        elif method == "Fill with Mode (Numeric)":
                            new_df = mode_filling(df, 0)
                        elif method == "Fill with Mode (Categorical)":
                            new_df = mode_filling(df, 1)
                        else:
                            new_df = interpolate_fill(df, 0)

                        st.session_state['df'] = new_df
                        st.success(f"Successfully applied '{method}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Null handling failed: {e}")

            st.divider()

            # Outlier detection
            st.subheader("3. Outlier Detection")
            st.write("Detect and handle outliers using the IQR (Interquartile Range) method.")
            numeric_cols = st.session_state['df'].select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                st.info("No numeric columns found for outlier detection.")
            else:
                if st.button("Handle Outliers (IQR Method)"):
                    try:
                        new_df = handle_outliers(st.session_state['df'])
                        st.session_state['df'] = new_df
                        st.success("Outliers have been clipped using the IQR method.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Outlier detection failed: {e}")

            st.divider()

            # Typecasting
            st.subheader("4. Typecasting")
            st.write("Convert column data types to ensure proper data format.")
            cast_cols = st.multiselect("Select columns to cast:", df.columns.tolist())
            new_type = st.selectbox("Select target data type:", ["int", "float", "str", "datetime64[ns]"])
            if st.button("Apply Type Cast"):
                try:
                    new_df = typecasting(df, cast_cols, new_type)
                    st.session_state['df'] = new_df
                    st.success(f"Columns '{cast_cols}' cast to {new_type}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to cast: {e}")
            
            st.divider()
            
            st.subheader("Current Data Preview")
            st.dataframe(st.session_state['df'].head())
            export_section(st.session_state['df'])
        else:
            st.warning("Please upload a file in **Step 1** to begin cleaning.")

    # -------------------------------------------------------------
    # PAGE 4: DATA TRANSFORMATION
    # -------------------------------------------------------------
    elif choice == pages[3]:
        st.title("🔄 Data Transformation")
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            st.write("Transform and encode your data using various techniques.")

            # Transformation Type Selection
            transform_type = st.radio("Select transformation technique:", 
                                     ["Label Encoding", "Normalization", "Scaling", "Discretization"])
            
            # LABEL ENCODING
            if transform_type == "Label Encoding":
                st.subheader("🏷️ Label Encoding")
                st.write("Convert categorical variables into numerical labels.")
                
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not categorical_cols:
                    st.warning("No categorical columns found for label encoding.")
                else:
                    selected_cols = st.multiselect("Select categorical columns to encode:", categorical_cols, default=categorical_cols[:1] if categorical_cols else [])
                    
                    if selected_cols and st.button("Apply Label Encoding"):
                        try:
                            new_df, mappings = label_encoding(df, selected_cols)
                            st.session_state['df'] = new_df
                            st.success(f"Label encoding applied to: {selected_cols}")
                            st.write("Encoding Mappings:")
                            for col, mapping in mappings.items():
                                st.write(f"**{col}**: {mapping}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Label encoding failed: {e}")
            
            # NORMALIZATION
            elif transform_type == "Normalization":
                st.subheader("📊 Normalization")
                st.write("Scale numeric values to a standard range [0,1] or standardized distribution.")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if not numeric_cols:
                    st.warning("No numeric columns found for normalization.")
                else:
                    selected_cols = st.multiselect("Select numeric columns to normalize:", numeric_cols, default=numeric_cols[:1] if numeric_cols else [])
                    norm_method = st.radio("Choose normalization method:", ["Min-Max (0-1 range)", "Z-Score (Standardization)"])
                    
                    if selected_cols and st.button("Apply Normalization"):
                        try:
                            if norm_method == "Min-Max (0-1 range)":
                                new_df = minmax(df, selected_cols)
                            else:  # Z-Score
                                new_df = zscore(df, selected_cols)
                            
                            st.session_state['df'] = new_df
                            st.success(f"Normalization applied to: {selected_cols}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Normalization failed: {e}")
            
            # SCALING
            elif transform_type == "Scaling":
                st.subheader("📈 Scaling")
                st.write("Scale numeric values using various scaling methods.")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if not numeric_cols:
                    st.warning("No numeric columns found for scaling.")
                else:
                    selected_cols = st.multiselect("Select numeric columns to scale:", numeric_cols, default=numeric_cols[:1] if numeric_cols else [])
                    scale_method = st.radio("Choose scaling method:", ["Min-Max Scaling", "Standard Scaling"])
                    
                    if selected_cols and st.button("Apply Scaling"):
                        try:
                            if scale_method == "Min-Max Scaling":
                                new_df = min_max_scaling(df, selected_cols)
                            else:  # Standard Scaling
                                new_df = standard_scaling(df, selected_cols)
                            
                            st.session_state['df'] = new_df
                            st.success(f"Scaling applied to: {selected_cols}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Scaling failed: {e}")
            
            # DISCRETIZATION
            elif transform_type == "Discretization":
                st.subheader("📦 Discretization")
                st.write("Convert continuous variables into discrete bins or categories.")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if not numeric_cols:
                    st.warning("No numeric columns found for discretization.")
                else:
                    selected_col = st.selectbox("Select numeric column to discretize:", numeric_cols)
                    disc_method = st.radio("Choose discretization method:", 
                                          ["Equal Width", "Equal Frequency", "K-Bins"])
                    
                    if selected_col:
                        if disc_method == "Equal Width":
                            n_bins = st.slider("Number of bins:", 2, 20, 5)
                            if st.button("Apply Equal Width Discretization"):
                                try:
                                    new_df = equal_width(df, selected_col, bin_number=n_bins)
                                    st.session_state['df'] = new_df
                                    st.success(f"Equal width discretization applied to '{selected_col}' with {n_bins} bins.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Discretization failed: {e}")
                        
                        elif disc_method == "Equal Frequency":
                            n_quantiles = st.slider("Number of quantiles:", 2, 20, 4)
                            if st.button("Apply Equal Frequency Discretization"):
                                try:
                                    new_df = equal_frequency(df, selected_col, quartile_number=n_quantiles)
                                    st.session_state['df'] = new_df
                                    st.success(f"Equal frequency discretization applied to '{selected_col}' with {n_quantiles} quantiles.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Discretization failed: {e}")
                        
                        elif disc_method == "K-Bins":
                            n_bins = st.slider("Number of bins:", 2, 20, 5)
                            bin_strategy = st.selectbox("Binning strategy:", ["quantile", "kmeans", "uniform"])
                            if st.button("Apply K-Bins Discretization"):
                                try:
                                    new_df = kbins(df, [selected_col], bin_number=n_bins, strat=bin_strategy)
                                    st.session_state['df'] = new_df
                                    st.success(f"K-Bins discretization applied to '{selected_col}' using {bin_strategy} strategy.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"K-Bins discretization failed: {e}")

            st.divider()

            st.subheader("Current Data Preview")
            st.dataframe(st.session_state['df'].head())
            export_section(st.session_state['df'])
        else:
            st.warning("Please upload a file in **Step 1** to begin transformation.")

    # -------------------------------------------------------------
    # PAGE 5: DATA REDUCTION
    # -------------------------------------------------------------
    elif choice == pages[4]:
        st.title("📉 Data Reduction")
        if st.session_state['df'] is not None:
            df = st.session_state['df']
            
            # Reduction Type Selection
            reduction_type = st.radio("Select data reduction technique:", 
                                     ["Dimensionality Reduction", "Numerosity Reduction", "Data Compression"])
            
            # DIMENSIONALITY REDUCTION
            if reduction_type == "Dimensionality Reduction":
                st.subheader("🔍 Dimensionality Reduction")
                st.write("Reduce the number of features while preserving important information.")
                
                dim_method = st.radio("Choose a dimensionality reduction method:", ["PCA", "tSNE"])
                
                # Get numeric columns only
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if not numeric_cols:
                    st.warning("No numeric columns found for dimensionality reduction.")
                else:
                    selected_cols = st.multiselect("Select numeric columns to reduce:", numeric_cols, default=numeric_cols)
                    
                    if dim_method == "PCA":
                        n_components = st.slider("Number of components:", 1, min(len(selected_cols), 10), 2)
                        if st.button("Apply PCA"):
                            try:
                                new_df = pca(df, selected_cols, n=n_components)
                                st.session_state['df'] = new_df
                                st.success(f"PCA applied! Reduced to {n_components} components.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"PCA failed: {e}")
                    
                    elif dim_method == "tSNE":
                        n_components = st.slider("Number of components:", 1, min(len(selected_cols), 3), 2)
                        perplexity = st.slider("Perplexity:", 5, min(30, len(df)//5), 30)
                        if st.button("Apply tSNE"):
                            try:
                                new_df = tsne(df, selected_cols, n=n_components, perp=perplexity)
                                st.session_state['df'] = new_df
                                st.success(f"tSNE applied! Reduced to {n_components} components.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"tSNE failed: {e}")
            
            # NUMEROSITY REDUCTION
            elif reduction_type == "Numerosity Reduction":
                st.subheader("📊 Numerosity Reduction")
                st.write("Reduce the number of records/rows in your dataset.")
                
                num_method = st.radio("Choose a numerosity reduction method:", 
                                     ["Random Sampling", "Stratified Sampling", "Cluster Reduction"])
                
                if num_method == "Random Sampling":
                    frac = st.slider("Fraction of data to keep:", 0.1, 1.0, 0.5, 0.1)
                    if st.button("Apply Random Sampling"):
                        try:
                            new_df = random_sampling(df, fract=frac)
                            st.session_state['df'] = new_df
                            st.success(f"Random sampling applied! Kept {int(frac*100)}% of rows.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Random sampling failed: {e}")
                
                elif num_method == "Stratified Sampling":
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    if not cat_cols:
                        st.warning("No categorical columns found for stratified sampling.")
                    else:
                        strat_col = st.selectbox("Select column to stratify by:", cat_cols)
                        frac = st.slider("Fraction of data to keep:", 0.1, 1.0, 0.5, 0.1)
                        if st.button("Apply Stratified Sampling"):
                            try:
                                new_df = stratified_sampling(df, strat_col, fract=frac)
                                st.session_state['df'] = new_df
                                st.success(f"Stratified sampling applied! Kept {int(frac*100)}% of rows.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Stratified sampling failed: {e}")
                
                elif num_method == "Cluster Reduction":
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if not numeric_cols:
                        st.warning("No numeric columns found for cluster reduction.")
                    else:
                        n_clusters = st.slider("Number of clusters:", 2, min(50, len(df)), 10)
                        if st.button("Apply Cluster Reduction"):
                            try:
                                new_df = cluster_reduction(df, n_clusters)
                                st.session_state['df'] = new_df
                                st.success(f"Cluster reduction applied! Reduced to {n_clusters} cluster centers.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Cluster reduction failed: {e}")
            
            # DATA COMPRESSION
            elif reduction_type == "Data Compression":
                st.subheader("🗜️ Data Compression")
                st.write("Reduce memory usage by optimizing data types.")
                
                comp_method = st.radio("Choose a compression method:", 
                                      ["Lossy Compression", "Lossless Compression"])
                
                if comp_method == "Lossy Compression":
                    st.info("Lossy compression may reduce precision but saves more memory. Converts dtypes intelligently.")
                    if st.button("Apply Lossy Compression"):
                        try:
                            mem_before = df.memory_usage(deep=True).sum() / 1024**2
                            new_df = lossy(df)
                            mem_after = new_df.memory_usage(deep=True).sum() / 1024**2
                            st.session_state['df'] = new_df
                            st.success(f"Lossy compression applied! Memory reduced from {mem_before:.2f}MB to {mem_after:.2f}MB.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lossy compression failed: {e}")
                
                elif comp_method == "Lossless Compression":
                    st.info("Lossless compression preserves all data while optimizing dtypes for storage.")
                    if st.button("Apply Lossless Compression"):
                        try:
                            mem_before = df.memory_usage(deep=True).sum() / 1024**2
                            new_df = lossless(df)
                            mem_after = new_df.memory_usage(deep=True).sum() / 1024**2
                            st.session_state['df'] = new_df
                            st.success(f"Lossless compression applied! Memory reduced from {mem_before:.2f}MB to {mem_after:.2f}MB.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lossless compression failed: {e}")

            st.divider()

            st.subheader("Current Data Preview")
            st.dataframe(st.session_state['df'].head())
            export_section(st.session_state['df'])
        else:
            st.warning("Please upload a file in **Step 1** to begin reduction.")

if __name__ == '__main__':
    main()
