import streamlit as st
import pickle
import pandas as pd

# List of teams and cities used in the model
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai', 
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# Load the pre-trained model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Title for the web app
st.title('IPL Win Predictor')

# Create UI elements for input selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

# Inputs for the match target, current score, overs, and wickets
target = st.number_input('Target')

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

# Button to trigger prediction
if st.button('Predict Probability'):
    # Ensure valid input for overs and wickets
    if overs == 0:
        st.error("Overs cannot be zero.")
    elif wickets > 10 or wickets < 0:
        st.error("Wickets must be between 0 and 10.")
    else:
        # Calculations for prediction input
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        remaining_wickets = 10 - wickets
        crr = score / overs if overs > 0 else 0  # Avoid division by zero
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Avoid division by zero

        # Creating input DataFrame for the model
        input_df = pd.DataFrame({'batting_team': [batting_team],
                                 'bowling_team': [bowling_team],
                                 'city': [selected_city],
                                 'runs_left': [runs_left],
                                 'balls_left': [balls_left],
                                 'wickets': [remaining_wickets],
                                 'total_runs_x': [target],
                                 'crr': [crr],
                                 'rrr': [rrr]})

        # Show the input dataframe to check if columns are correctly formatted
        st.write("Input DataFrame:", input_df)

        # Make the prediction using the model pipeline
        try:
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]
            
            # Display the results
            st.header(batting_team + ": " + str(round(win * 100)) + "% chance of winning")
            st.header(bowling_team + ": " + str(round(loss * 100)) + "% chance of winning")
        except Exception as e:
            # Handle any errors during prediction
            st.error(f"Error occurred during prediction: {e}")
