
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

filename = 'lin_reg.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('Penguin Species Prediction')
st.subheader('Enter the data')

data = sns.load_dataset('penguins')
data.to_csv('penguins')

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    prediction = loaded_model.predict(data)
    st.write(prediction)