import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import KNNImputer

import os

import geopandas as gpd

import folium
from IPython.display import display, HTML

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

import json
import requests

st.cache_data(show_spinner=False)
def load_data():
    # Import the data and clean it
    df = pd.read_excel('datasciencelab.xlsx')

    # separate the numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_data = df[numeric_columns]

    # Standardize the data
    scaler = StandardScaler()
    numeric_data_scaled = scaler.fit_transform(numeric_data)

    # Apply the KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    numeric_data_imputed = imputer.fit_transform(numeric_data_scaled)

    # Inverse the scaling and store the result in the DataFrame
    df[numeric_columns] = scaler.inverse_transform(numeric_data_imputed)

    eu_mapping = {
        "AT": "Austria",
        "BE": "Belgium",
        "BG": "Bulgaria",
        "HR": "Croatia",
        "CY": "Cyprus",
        "CZ": "Czech Republic",
        "DK": "Denmark",
        "EE": "Estonia",
        "FI": "Finland",
        "FR": "France",
        "DE": "Germany",
        "EL": "Greek",        
        "HU": "Hungary",
        "IE": "Ireland",
        "IT": "Italy",
        "LV": "Latvia",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "MT": "Malta",
        "NL": "Netherlands",
        "PL": "Poland",
        "PT": "Portugal",
        "RO": "Romania",
        "SK": "Slovakia",
        "SI": "Slovenia",
        "ES": "Spain",
        "SE": "Sweden"
    }

    df['nation_code'] = df['NUTS'].str[:2]
    df['Nation'] = df['nation_code'].map(eu_mapping)

    # Normalizzazione 
    df_numeric = df.select_dtypes(include=[np.number])
    df_norm = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())

    # Creazione indice di prospettive di vita
    weights = {'Healthy life expectancy': 0.35, 'Road transport performance': 0.2, 'Rail transport performance': 0.2, 'Suicide death rate': -0.25}
    df['Life_prospects'] = sum(df_norm[col] * weight for col, weight in weights.items())

    # Creazione indice di prospettive di carriera
    weights = {'Unemployment rate': -0.2, 'Employment rate (excluding agriculture)': 0.2, 'NEET rate': -0.3, 'Disposable income per capita': 0.3}
    df['Career_prospects'] = sum(df_norm[col] * weight for col, weight in weights.items())
    return df

df = load_data()

st.title('Career and Life Prospects in Europe: The Ideal Destination for Young Graduates')

# crea due slidere (Life prospects e Career prospects) per permettere all'utente di selezionare i due indicatori
life_prospects = st.slider('Life prospects', 0, 100, 50)
career_prospects = st.slider('Career prospects', 0, 100, 50)
if life_prospects == 0 and career_prospects == 0:
    st.write(':red[Please select at least one indicator]')

# crea un grafico a barre sulla prosperità della vita da inserire su streamlit

df['Indice_composito'] = life_prospects*df['Life_prospects'] + career_prospects*df['Career_prospects']
Indice = df.groupby('Nation')['Indice_composito'].mean().sort_values(ascending=False)
st.write(Indice)

top_3 = Indice.head(3).index
st.write(f'The best country for you is {top_3[0]}, followed by {top_3[1]} and {top_3[2]}')

fig, ax = plt.subplots()
sns.barplot(x='Nation', y='Indice_composito', data=df, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

