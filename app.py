import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and preprocessing objects
try:
    with open('pipe.pkl', 'rb') as file:
        pipe = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Set title for the app
st.title('IPL Win Predictor')

# Teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Input columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

# Target input
target = st.number_input('Target', min_value=0, step=1)

# Score, overs, and wickets input
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets fallen', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('Predict Probability'):
    # Input validation
    if batting_team == bowling_team:
        st.error("Batting and bowling teams cannot be the same!")
    elif overs < 0 or overs > 20:
        st.error("Overs completed must be between 0 and 20.")
    elif target <= score:
        st.error("Target must be greater than the current score.")
    elif overs == 0:
        st.error("Overs completed cannot be zero.")
    else:
        try:
            # Calculate required parameters
            runs_left = target - score
            balls_left = 120 - int(overs * 6)
            remaining_wickets = 10 - wickets
            current_run_rate = score / overs if overs > 0 else 0
            required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else float('inf')

            # Create input DataFrame
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [remaining_wickets],
                'total_runs_x': [target],
                'crr': [current_run_rate],
                'rrr': [required_run_rate]
            })

            # Make prediction
            result = pipe.predict_proba(input_df)
            
            # Display results
            win_prob = result[0][1] * 100
            loss_prob = result[0][0] * 100
            
            # Create success message box
            st.success(f"""
            Prediction Results:
            - {batting_team} Win Probability: {win_prob:.1f}%
            - {bowling_team} Win Probability: {loss_prob:.1f}%
            """)

            # Add visualization
            import plotly.graph_objects as go
            
            fig = go.Figure(go.Bar(
                x=['Batting Team', 'Bowling Team'],
                y=[win_prob, loss_prob],
                text=[f'{win_prob:.1f}%', f'{loss_prob:.1f}%'],
                textposition='auto',
            ))
            
            fig.update_layout(
                title='Win Probability Distribution',
                yaxis_title='Probability (%)',
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Please ensure your model pipeline includes all necessary preprocessing steps.")
