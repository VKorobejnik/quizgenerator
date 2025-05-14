import streamlit as st
st.set_page_config(page_title="Quiz Generator") 

from core import login, quiz

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    
#st.write(f"Running file: {os.path.basename(__file__)}")

if not st.session_state.logged_in:
    login.login_page()
else:
    quiz.show()
    