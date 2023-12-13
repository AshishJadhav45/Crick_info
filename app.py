import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('IPL Win Predictor')

col1, col2 = st.beta_columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.beta_columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets_out = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    remaining_wickets = 10 - wickets_out  # Changed variable name to avoid confusion
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Create a sample pipeline for feature transformation
    sample_pipeline = Pipeline([
        ('column_transformer', ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), ['batting_team', 'bowling_team', 'city'])
            ],
            remainder='passthrough'
        ))
    ])

    # Use the actual input_df for transformation
    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left],
                             'wickets': [remaining_wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    # Transform the input data using the trained ColumnTransformer
    input_df_transformed = sample_pipeline.named_steps['column_transformer'].transform(input_df)

    # Perform prediction
    result = pipe.predict_proba(input_df_transformed)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")
