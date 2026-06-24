import streamlit as st
import pandas as pd
import io

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
        3. **Clean** the dataset by handling nulls, outliers, and duplicates.
        4. **Transform** variables via scaling or encodings.
        5. **Reduce** the dataset size via dropping columns.
        
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
            
            # Duplicates
            st.subheader("1. Handle Duplicates")
            duplicate_count = df.duplicated().sum()
            st.write(f"Duplicated rows found: **{duplicate_count}**")
            if duplicate_count > 0:
                if st.button("Drop Duplicate Rows"):
                    st.session_state['df'] = df.drop_duplicates()
                    st.success(f"Removed {duplicate_count} duplicate rows.")
                    st.rerun()

            st.divider()

            # Missing Values
            st.subheader("2. Handle Missing Values")
            missing_cols = df.columns[df.isnull().any()].tolist()
            if not missing_cols:
                st.write("✅ Great! No missing values detected in the dataset.")
            else:
                target_col = st.selectbox("Select column with missing values:", missing_cols)
                if target_col:
                    method = st.radio("Choose a strategy to handle missing values:", [
                        "Drop Rows with Nulls", 
                        "Fill with Mean (Numeric only)", 
                        "Fill with Median (Numeric only)", 
                        "Fill with Mode",
                        "Fill with Value '0'"
                    ])
                    
                    if st.button("Apply Missing Value Fix"):
                        new_df = df.copy()
                        try:
                            if method == "Drop Rows with Nulls":
                                new_df = new_df.dropna(subset=[target_col])
                            elif method == "Fill with Mean (Numeric only)":
                                new_df[target_col] = new_df[target_col].fillna(new_df[target_col].mean())
                            elif method == "Fill with Median (Numeric only)":
                                new_df[target_col] = new_df[target_col].fillna(new_df[target_col].median())
                            elif method == "Fill with Mode":
                                new_df[target_col] = new_df[target_col].fillna(new_df[target_col].mode()[0])
                            elif method == "Fill with Value '0'":
                                new_df[target_col] = new_df[target_col].fillna(0)
                                
                            st.session_state['df'] = new_df
                            st.success(f"Successfully applied '{method}' to column '{target_col}'.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Operation failed. Ensure the strategy matches the data type. Error: {e}")

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
            st.write("Convert or scale data formats.")

            st.subheader("1. Typecasting")
            cast_col = st.selectbox("Select column to cast:", df.columns.tolist())
            new_type = st.selectbox("Select target data type:", ["int", "float", "object", "datetime64[ns]"])
            if st.button("Apply Type Cast"):
                new_df = df.copy()
                try:
                    if new_type == "datetime64[ns]":
                        new_df[cast_col] = pd.to_datetime(new_df[cast_col])
                    else:
                        new_df[cast_col] = new_df[cast_col].astype(new_type)
                    st.session_state['df'] = new_df
                    st.success(f"Column '{cast_col}' cast to {new_type}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to cast: {e}")

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
            
            st.subheader("Drop Columns")
            cols_to_drop = st.multiselect("Select columns you want to remove:", df.columns.tolist())
            if st.button("Drop Selected Columns"):
                if cols_to_drop:
                    st.session_state['df'] = df.drop(columns=cols_to_drop)
                    st.success(f"Dropped columns: {cols_to_drop}")
                    st.rerun()

            st.divider()

            st.subheader("Current Data Preview")
            st.dataframe(st.session_state['df'].head())
            export_section(st.session_state['df'])
        else:
            st.warning("Please upload a file in **Step 1** to begin reduction.")

if __name__ == '__main__':
    main()
