import streamlit as st
import pandas as pd
import polars as pl
from ydata_profiling import ProfileReport
import tempfile
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. Data Loading & Caching ---
# Use Streamlit's caching to avoid reloading data on every user interaction
@st.cache_data
def load_data(uploaded_file):
    """
    Loads data using Polars for efficiency with large files.
    """
    try:
        # Use lazy loading with Polars for large files
        df_pl = pl.read_csv(uploaded_file).lazy()
        # Collect the Polars LazyFrame to get a standard DataFrame
        # The app logic will operate on this collected DF
        df = df_pl.collect()
        return df.to_pandas()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- 2. Data Cleaning Operations ---
def perform_cleaning(df, operation):
    """
    Performs selected cleaning operation on the DataFrame.
    """
    if df is None:
        return pd.DataFrame()

    if operation == "Drop NaNs":
        return df.dropna()
    elif operation == "Remove Duplicates":
        return df.drop_duplicates()
    elif operation == "Normalize Numerical Columns":
        st.info("Normalization applied to all numerical columns.")
        df_numeric = df.select_dtypes(include=['number'])
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns, index=df_numeric.index)
        df_cleaned = df.copy()
        for col in df_numeric.columns:
            df_cleaned[col] = df_scaled[col]
        return df_cleaned
    else:
        return df

# --- 3. ML Model (Simulated) ---
# Create a dummy model and save it for loading
def create_and_save_dummy_model():
    """
    Creates a simple pre-trained model for demonstration.
    """
    # Create a small, dummy dataset for a classification task
    data = {
        'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature2': [100, 80, 60, 40, 20, 10, 30, 50, 70, 90],
        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)

    X = df[['feature1', 'feature2']]
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    # Save the model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)

    return 'model.pkl'

# Load the dummy model (use st.cache_resource for heavy models)
@st.cache_resource
def load_ml_model():
    if not os.path.exists('model.pkl'):
        create_and_save_dummy_model()
    
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler']

# --- 4. Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide")
    st.title("Data Wizard 🧙‍♂️")
    st.markdown("A browser-based tool for Data Cleaning, Profiling & ML Prediction.")

    # Sidebar for File Upload
    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    # Session state to keep track of the DataFrame
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
    
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.sidebar.success("File loaded successfully!")
            
    # Conditional UI based on data availability
    if not st.session_state.df.empty:
        st.sidebar.header("2. Clean Data")
        cleaning_ops = ["None", "Drop NaNs", "Remove Duplicates", "Normalize Numerical Columns"]
        selected_op = st.sidebar.selectbox("Choose a cleaning operation:", cleaning_ops)

        if st.sidebar.button("Apply Cleaning"):
            with st.spinner("Applying cleaning operations..."):
                st.session_state.df = perform_cleaning(st.session_state.df.copy(), selected_op)
            st.success("Cleaning applied!")
            
        # Download button for cleaned data
        csv_file = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Cleaned Data as CSV",
            data=csv_file,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

        st.sidebar.header("3. Data Profiling")
        if st.sidebar.button("Generate Profile Report"):
            with st.spinner("Generating comprehensive report..."):
                profile = ProfileReport(st.session_state.df, explorative=True, dark_mode=True)
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
                    profile.to_file(tmp_file.name)
                    st.session_state.profile_path = tmp_file.name
            st.success("Report generated!")

        st.header("Data Preview")
        st.dataframe(st.session_state.df)

        if "profile_path" in st.session_state:
            # Display the report in an iframe for a seamless experience
            st.components.v1.html(
                open(st.session_state.profile_path, 'r').read(),
                height=600,
                scrolling=True
            )
            
        # --- ML Prediction Module ---
        st.sidebar.header("4. ML Prediction")
        with st.expander("Try a Simple Prediction"):
            model, scaler = load_ml_model()

            st.markdown("This module uses a simple pre-trained model to make a prediction.")
            st.markdown("Enter values for **feature1** and **feature2** to get a result.")
            
            col1, col2 = st.columns(2)
            with col1:
                feature1_val = st.number_input("Enter a value for Feature 1", value=0.0)
            with col2:
                feature2_val = st.number_input("Enter a value for Feature 2", value=0.0)
            
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    user_data = pd.DataFrame([[feature1_val, feature2_val]], columns=['feature1', 'feature2'])
                    user_data_scaled = scaler.transform(user_data)
                    prediction = model.predict(user_data_scaled)[0]
                    
                    st.success("Prediction complete!")
                    st.markdown(f"The model's prediction is: **{'Class 1' if prediction == 1 else 'Class 0'}**")

# Run the app
if __name__ == "__main__":
    main()