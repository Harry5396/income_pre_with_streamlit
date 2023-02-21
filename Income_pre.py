import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# App title and description
st.title('Adult Census Income')
st.markdown("""
Predict whether income exceeds $50K/yr based on census data.
- Type of Problem : Classification
- Business Value : Understand the standard of living of a region
""")
# https://www.kaggle.com/datasets/uciml/adult-census-income

st.code('" This model trained with: Accuracy: 92.15% "')


# Read data
@st.cache
def load_data():
    return pd.read_csv('final_data.csv')

data_load_state = st.text('Loading data..')
df = load_data()
df = df.drop(['Unnamed: 0'], axis=1)
data_load_state.text('Loading data.. done!')

# show 100 data example
st.subheader('Data')
st.dataframe(df[:100])


# Define components for the sidebar
st.sidebar.header('Input Features')
age = st.sidebar.slider(
    label='Age of a citizen',
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=int(df['age'].mean()),
    step=1)
fnlwgt = st.sidebar.slider(
    label='Final weight of a citizen',
    min_value=int(df['fnlwgt'].min()),
    max_value=int(df['fnlwgt'].max()),
    value=int(df['fnlwgt'].mean()),
    step=10)
education_num = st.sidebar.slider(
    label='Number of years for education of a citizen',
    min_value=int(df['education.num'].min()),
    max_value=int(df['education.num'].max()),
    value=int(df['education.num'].mean()),
    step=1)
hours_per_week = st.sidebar.slider(
    label='Working hours in a week',
    min_value=int(df['hours.per.week'].min()),
    max_value=int(df['hours.per.week'].max()),
    value=int(df['hours.per.week'].mean()),
    step=1)
marital_status= st.sidebar.selectbox(
    label='Status of marriage', 
    options=df['marital.status'].unique())
occupation= st.sidebar.selectbox(
    label='Occupation of a citizen', 
    options=df['occupation'].unique(),)
relationship= st.sidebar.selectbox(
    label='Relationship or role in family', 
    options=df['relationship'].unique())

st.subheader('With your choices:')

# col1, col2= st.columns(2)
# with col1:
#     st.caption('Age of a citizen:')
#     st.caption('Final weight of a citizen:')
#     st.caption('Number of years for education of a citizen:')
#     st.caption('Working hours in a week:')
#     st.caption('Status of marriage:')
#     st.caption('Occupation of a citizen:')
#     st.caption('Relationship or role in family:')

# with col2:
#     st.caption(age)
#     st.caption(fnlwgt)
#     st.caption(education_num)
#     st.caption(hours_per_week)
#     st.caption(marital_status)
#     st.caption(occupation)
#     st.caption(relationship)


# Input preprocessing
@st.cache
def FetureEngineering(df):
    dataset = df
    encoderSeries = {} #store encoder for invert
    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            encoder = LabelEncoder()
            dataset[col] = encoder.fit_transform(dataset[col])
            encoderSeries[col] = encoder

    scalerSeries = {}
    for col in dataset.columns:
        scaler = StandardScaler()
        dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1))
        scalerSeries[col] = scaler

    return encoderSeries, scalerSeries

encoderSeries, scalerSeries = FetureEngineering(df=df)

# Input
marital_status=encoderSeries['marital.status'].transform([marital_status])
occupation=encoderSeries['occupation'].transform([occupation])
relationship=encoderSeries['relationship'].transform([relationship])

# print('abcd:', marital_status, occupation, relationship)

age=(age - scalerSeries['age'].mean_)/scalerSeries['age'].scale_
fnlwgt=(fnlwgt - scalerSeries['fnlwgt'].mean_)/scalerSeries['fnlwgt'].scale_
education_num=(education_num - scalerSeries['education.num'].mean_)/scalerSeries['education.num'].scale_
hours_per_week=(hours_per_week - scalerSeries['hours.per.week'].mean_)/scalerSeries['hours.per.week'].scale_
marital_status=(marital_status - scalerSeries['marital.status'].mean_)/scalerSeries['marital.status'].scale_
occupation=(occupation - scalerSeries['occupation'].mean_)/scalerSeries['occupation'].scale_
relationship=(relationship - scalerSeries['relationship'].mean_)/scalerSeries['relationship'].scale_


input_ = [age, fnlwgt, education_num, hours_per_week, marital_status, occupation, relationship]
input_ = np.array(input_).reshape(1, -1)


# Load model
model = pickle.load(open("income_pre.pkl", "rb"))
output = model.predict(input_)
# print(output)

st.code('AI predicted your income:')
if output[0]==1:
    st.subheader('It could be :blue[higher] than 50K! :sunglasses:')
else:
    st.subheader('It could be :blue[lower] than 50K! :sunglasses:')
