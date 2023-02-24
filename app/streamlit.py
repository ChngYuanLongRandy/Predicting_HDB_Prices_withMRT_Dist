import streamlit as st
from src.datapipeline import Modelpipeline
from src.fetch_resources import fetch_resources
import pandas as pd
import logging


def main():
    test_file_path = "data/hdb_latest.csv"
    test_data_pd = pd.read_csv(test_file_path)

    st.title('HDB Resale Price Prediction')

    test_HDB_storey_range = sorted(test_data_pd.storey_range.unique().tolist())
    test_HDB_flat_type = sorted(test_data_pd.flat_type.unique().tolist())

    test_HDB_flat_type.remove('MULTI-GENERATION')

    postal_code = st.number_input('Postal Code', 560325)

    storey_range = st.selectbox('Storey Range', test_HDB_storey_range)

    flat_type = st.selectbox('Flat Type', test_HDB_flat_type)

    button_click = st.button('Predict Value of Flat Today')

    input = {
        'postal_code': 560325,
        'storey_range':storey_range,
        'flat_type':flat_type
        }

    info = fetch_resources(input)

    if button_click:
        st.write('Prediction value here')

if __name__ == "__main__":
    main()