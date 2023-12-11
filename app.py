import os
import pickle
import streamlit as st
import pandas as pd


def get_pipe_path():
    """
    Returns the absolute path to the 'pipe.pkl' file.
    Handles potential errors.
    """
    script_path = os.path.abspath(__file__)
    try:
        # Check if the file exists in the same directory
        if os.path.exists(os.path.join(os.path.dirname(script_path), 'pipe.pkl')):
            return os.path.join(os.path.dirname(script_path), 'pipe.pkl')
        # If not, try checking another location (e.g., 'Crick_info' directory)
        elif os.path.exists('Crick_info/pipe.pkl'):
            return 'Crick_info/pipe.pkl'
        else:
            raise FileNotFoundError("File 'pipe.pkl' not found")
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.stop()


# Get the pipe path using the function
try:
    file_path = get_pipe_path()
except Exception:
    # Handle any unexpected errors
    st.error("An unexpected error occurred. Please try again later.")
    st.stop()

# Load the model
try:
    with open(file_path, 'rb') as file:
        pipe = pickle.load(file)
except Exception as e:
    st.error(f"Error loading pickled object: {e}")
    st.stop()

# Define teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali',]

# Streamlit UI
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
