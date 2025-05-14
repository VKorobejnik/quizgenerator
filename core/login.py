import streamlit as st
import json
import base64
import bcrypt
import time


USER_DATA_FILE = "user_data.b64"

def load_user_data():
    try:
        with open(USER_DATA_FILE, "r") as f:
            encoded = f.read()
            decoded = base64.b64decode(encoded.encode()).decode()
            return json.loads(decoded)
    except:
        return {}

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

# Login UI
def login_page():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        #st.markdown("Quiz Generator Login")
        with st.container(border=True):
            username = st.text_input("Username", key=f"username_input")
            password = st.text_input("Password", type="password", key=f"password_input")

            if st.button("Login"):
                    user_data = load_user_data()
                    if username in user_data and verify_password(password, user_data[username]):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please contact your system administrator.")

