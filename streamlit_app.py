import streamlit as st
import requests
from dotenv import load_dotenv
import os
from datetime import datetime,timedelta
import pandas as pd
from modules import utils
import numpy as np
from openai import OpenAI
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np


load_dotenv()

is_logged_in = False

if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False


# Helper function to format the selectbox options for places
def format_select_option(pair):
    return f"{pair[0]} ({pair[1]})"

def login():
    # Set background image
    # st.markdown(f'<style>body{{background-image: url({page_bg}); background-size: cover;}}</style>', unsafe_allow_html=True)
    global is_logged_in
    st.title('CineSphere')
    st.subheader('Welcome to CineSphere! Please Login to proceed.')
    # Get user input
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        # Check if login is valid
        data = {
                "grant_type": "password",
                "username": email,
                "password": password
                }
        
        #check if access token required as such
        if utils.get_user(email, password):
           
            st.success("Logged in!")
            
            st.session_state['is_logged_in'] = True
            st.session_state['email'] = email
            st.session_state['password'] = password
         
        else:
            st.error("Incorrect email or password")

def signup():
    # Set background image
    # st.markdown(f'<style>body{{background-image: url({page_bg}); background-size: cover;}}</style>', unsafe_allow_html=True)

    st.subheader('Signup')
    # Get user input
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    create_account = st.button("Create Account")
    
    if "create_account_state" not in st.session_state:
        st.session_state.create_account_state = False

    # Signup button
    if create_account:
        # Check if password matches
        if password != confirm_password:
            st.error("Passwords do not match")
        else:
            if not utils.check_user(email):
                utils.create_user(email, password)
                if utils.get_user(email, password) :
                    st.success("Signed up! Please complete the onboarding process below.")
                    st.session_state.create_account_state = True
                else:
                    st.error("Error signing up")
            else:
                st.error("Email already exists")

def home_page():
    # Set background image
    # st.markdown(f'<style>body{{background-image: url({home_bg}); background-size: cover;}}</style>', unsafe_allow_html=True)
    st.markdown("# CineSphere")

    # Create a menu with the options
    # menu = ["Home", "Login", "Signup"]
    # choice = st.sidebar.selectbox("Select an option", menu)

    # if choice == "Login":
        # login()
    # elif choice == "Signup":
        # signup()


def chat_interface_page():
    # Set background image
    # st.markdown(f'<style>body{{background-image: url({page_bg}); background-size: cover;}}</style>', unsafe_allow_html=True)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    st.markdown("# CineSphere")
    st.subheader('Have a question? Ask us anything! 🎥')
    st.text('Just so you know, we can answer any questions related to movies,\n give recommendations based on movies, actors and even the plot! ')

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        response = utils.chat_bot(prompt)   
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()     



def onboarding_page():
    
    st.markdown("# CineSphere")
    st.subheader('Onboarding! 🎥')
    st.text('Please select your favorite movies to get started! 🍿🎬🎥')
    
    df = utils.get_sample_movies()
    df['posterLink'] = df['movieId'].apply(lambda x : utils.getImage(x))
    
    num_cols = 5
    cols = st.columns(num_cols)
    
    selected_movies = []
    selected_movie_ids = []
    
    with st.form('movie_selection_form'):
        for i, (_, row) in enumerate(df.iterrows()):
            with cols[i % 5]:
                st.image(row['posterLink'], use_column_width=True)
                if st.checkbox(f"{row['movieName']}", key=f"checkbox_{row['movieId']}"):
                    selected_movies.append(row['movieName'])
                    selected_movie_ids.append(row['movieId'])

        st.write("You selected the following movies:")
        for movie in selected_movies:
            st.write(f"- {movie}")
        
        # Display the selected movies
        if st.form_submit_button("Submit"):
            utils.save_preferences(selected_movie_ids,st.session_state['email'],st.session_state['password'])
            st.write("Preferences saved successfully! Check your recommendations on the next page! 🎉")

def recommendations_page():
    
    st.markdown("# CineSphere")
    st.subheader('Recommendations! 🎥')
    st.text('Here are some movies you might like! 🍿')
    
    similarity = utils.generate_recommendations(st.session_state['email'],st.session_state['password'])
    # Split the DataFrame into buckets of size 10
    buckets = np.array_split(similarity, len(similarity) // 10)

    if 'current_bucket_index' not in st.session_state:
        st.session_state.current_bucket_index = 0
    
    current_bucket = buckets[st.session_state.current_bucket_index]
        
    cols = st.columns(5)
    for i, (_, row) in enumerate(current_bucket.iterrows()):
        with cols[i % 5]:
            st.image(row['posterLink'], use_column_width=True)
            st.write(row['Movie2'])
    
    # Add a "Refresh" button to show the next bucket
    if st.button("Refresh"):
        st.session_state.current_bucket_index = (st.session_state.current_bucket_index + 1) % len(buckets)   
        
pages = {
        "Home": home_page,
        "Question? Chat it out": chat_interface_page,
        "Recommendations": recommendations_page,
        "Onboarding": onboarding_page,
    }
        
# Define the Streamlit app
def main():
    st.set_page_config(
        page_title="CineSphere",page_icon=":popcorn:" ,layout="wide"
    )
    st.sidebar.title("Navigation")

    # Render the navigation sidebar
    if st.session_state['is_logged_in']==True:
        selection = st.sidebar.radio(
            "Go to", ["Home","Onboarding","Recommendations","Question? Chat it out","Log Out"]
        )
    else:
        selection = st.sidebar.radio("Go to", ["Sign In", "Sign Up"])

    # Render the selected page or perform logout
    if selection == "Log Out":
        st.session_state.clear()
        st.sidebar.success("You have successfully logged out!")
        st.experimental_rerun()
    elif selection == "Sign In":
        token = login()
        if token is not None:
            st.session_state.token = token
            st.experimental_rerun()
    elif selection == "Sign Up":
        signup()
    else:
        page = pages[selection]
        page()


if __name__ == "__main__":
    main()        