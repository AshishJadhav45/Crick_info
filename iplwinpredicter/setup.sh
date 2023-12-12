
# Install scikit-learn
pip install scikit-learn

# Create Streamlit configuration directory if it doesn't exist
mkdir -p ~/.streamlit/

# Create Streamlit configuration file with specified settings
echo "\
[server]\n\
port = \$PORT\n\
enableCORS = false\n\
headless = true\n\
" > ~/.streamlit/config.toml
