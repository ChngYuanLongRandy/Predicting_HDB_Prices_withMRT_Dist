import streamlit as st
from src.datapipeline import Modelpipeline
from src.fetch_resources import fetch_resources
import pandas as pd
import logging
from app.api.config import SETTINGS


def main():
    st.title('HDB Resale Price Prediction')

    test_HDB_storey_range = sorted(dataset.storey_range.unique().tolist())
    test_HDB_flat_type = sorted(dataset.flat_type.unique().tolist())

    test_HDB_flat_type.remove('MULTI-GENERATION')

    postal_code = st.number_input('Postal Code', 560325)

    storey_range = st.selectbox('Storey Range', test_HDB_storey_range)

    flat_type = st.selectbox('Flat Type', test_HDB_flat_type)

    button_click = st.button('Predict Value of Flat Today')

    input = {
        'postal_code': postal_code,
        'storey_range':storey_range,
        'flat_type':flat_type
        }

    complete_input = fetch_resources(input)

    if button_click:
        if complete_input is not None:
            prediction = prediction_pipe.predict(complete_input)
            st.write(prediction)
        else:
            st.write('Details not found, please enter a valid input')


if __name__ == "__main__":
    dataset = pd.read_csv(SETTINGS.dict()['DATA_PATH'])
    prediction_pipe = Modelpipeline(dataset)
    prediction_pipe.preprocess(training_size=0.5, split=False)
    prediction_pipe.load_model(SETTINGS.dict()['MODEL_PATH'])
    main()