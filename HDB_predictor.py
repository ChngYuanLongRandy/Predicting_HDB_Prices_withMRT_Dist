#! Python3
# Runs the script on streamlit

import streamlit as st
import numpy as np
import pandas as pd
import requests

github_repo = 'https://raw.githubusercontent.com/ChngYuanLongRandy/Predicting_HDB_Prices_withMRT_Dist/main/HDBdata%20with%20district_coord_dist_full.csv'

df = pd.read_csv(github_repo)


st.title("Predicting HDB Prices")

st.write('''
Select House Details below:
''')

st_street_name = st.selectbox("Select Address", sorted(set(df.street_name)))

df_refined = df[df.street_name == st_street_name]

# only show applicable blocks from the street name
st_block = st.selectbox('Block Number' , sorted(set(df_refined.block)))

df_refined = df[(df.street_name == st_street_name) & (df.block == st_block)]

# only show applicable blocks from the street name
st_storey_range = st.selectbox('Storey Range' , sorted(set(df_refined.storey_range)))

# only show applicable blocks from the street name
st_flat_type = st.selectbox("Select Flat Type", sorted(set(df_refined.flat_type)))

# # only show applicable blocks from the street name
# st_age_of_flat = st.selectbox('Storey Range' , sorted(set(df.storey_range)))

year_list = list(range(2022,2030,1))

st.slider('To which year?',
          min_value=2022,
          max_value=2030,
          step=1,
          )

st.button('Predict')