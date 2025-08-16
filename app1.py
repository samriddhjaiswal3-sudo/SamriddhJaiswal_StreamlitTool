import streamlit as st
import pandas as pd
import polars as pl

st.title('Simple Data App')
st.write("This app is a working test version.")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())
