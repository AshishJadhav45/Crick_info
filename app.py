import streamlit as st
import pickle
import pandas as pd

# Load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set title for the app
st.title('IPL Win Predictor')

# Teams and cities
teams = sorted([
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
])

cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
])

# Input columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', teams)
with col2:
    bowling_team = st.selectbox('Select the bowling team', teams)

selected_city = st.selectbox('Select host city', cities)

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
    # Validate inputs
    if overs < 0 or overs > 20:
        st.error("Overs completed must be between 0 and 20.")
    elif target < score:
        st.error("Target must be greater than the current score.")
    elif overs == 0:
        st.error("Overs completed cannot be zero.")
    else:
        # Calculate the necessary parameters
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        remaining_wickets = 10 - wickets
        current_run_rate = score / overs if overs > 0 else 0
        required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Prepare input DataFrame for the model
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

        # Ensure categorical features are in the correct format
        input_df['batting_team'] = input_df['batting_team'].astype('category')
        input_df['bowling_team'] = input_df['bowling_team'].astype('category')
        input_df['city'] = input_df['city'].astype('category')

        # Check for missing values
        if input_df.isnull().sum().sum() > 0:
            st.error("There are missing values in the input data.")
        else:
            # Predict the probabilities
            try:
                result = pipe.predict_proba(input_df)

                # Extract probabilities for win/loss
                loss_prob = result[0][0]
                win_prob = result[0][1]

                # Display the result
                st.subheader(f"Win Probability for {batting_team}: {win_prob * 100:.2f}%")
                st.subheader(f"Win Probability for {bowling_team}: {loss_prob * 100:.2f}%")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
