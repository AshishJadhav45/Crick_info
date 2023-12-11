import os
import pickle
import streamlit as st
import pandas as pd

def get_pipe_path():
    # Construct the path to pipe.pkl in the root directory
    pipe_path = os.path.join(os.getcwd(), 'Crick_info', 'pipe.pkl')

    return pipe_path

# Use the function to get the path to 'pipe.pkl'
file_path = get_pipe_path()

# Convert the file path to lowercase for case-insensitive comparison
lowercase_file_path = file_path.lower()

# Check if the file exists
if os.path.exists(lowercase_file_path):
    # Load the pickled object
    try:
        with open(lowercase_file_path, 'rb') as file:
            pipe = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading pickled object: {e}")
        st.stop()
else:
    st.error(f"Error: File '{file_path}' not found.")
    # You might want to handle this error appropriately, for example, by exiting the script or providing a default object.

# ... (rest of your code remains unchanged)

    # You might want to handle this error appropriately, for example, by exiting the script or providing a default object.

# ... (rest of your code remains unchanged)

    # You might want to handle this error appropriately, for example, by exiting the script or providing a default object.

    
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali',]

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
