import streamlit as st
import json
import base64
import bcrypt
import os

# File to store user data (Base64-encoded JSON)
USER_DATA_FILE = "user_credentials.b64"

# Initialize user data file if it doesn't exist
def init_user_data():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "w") as f:
            f.write(base64.b64encode(json.dumps({}).encode()).decode())

# Load user data (Base64-decoded)
def load_user_data():
    try:
        with open(USER_DATA_FILE, "r") as f:
            encoded_data = f.read()
            decoded_data = base64.b64decode(encoded_data.encode()).decode()
            return json.loads(decoded_data)
    except:
        return {}

# Save user data (Base64-encoded)
def save_user_data(data):
    encoded_data = base64.b64encode(json.dumps(data).encode()).decode()
    with open(USER_DATA_FILE, "w") as f:
        f.write(encoded_data)

# Hash a password
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

# Verify a password
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

# Register a new user
def register_user(username, password):
    user_data = load_user_data()
    if username in user_data:
        return False  # User already exists
    user_data[username] = hash_password(password)
    save_user_data(user_data)
    return True

# Authenticate a user
def authenticate_user(username, password):
    user_data = load_user_data()
    if username in user_data and verify_password(password, user_data[username]):
        return True
    return False

# Streamlit UI
def main():
    st.title("🔒 Secure Login System")

    # Initialize user data
    #init_user_data()

    # Login or Register Tabs
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.success("✅ Login successful!")
                st.session_state.logged_in = True
                st.session_state.username = username
            else:
                st.error("❌ Invalid username or password")

    with tab2:
        st.subheader("Register")
        new_username = st.text_input("New Username", key="reg_username")
        new_password = st.text_input("New Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("❌ Passwords do not match!")
            elif register_user(new_username, new_password):
                st.success("✅ Registration successful! Please login.")
            else:
                st.error("❌ Username already exists!")

    # Display logged-in state
    if st.session_state.get("logged_in"):
        st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

if __name__ == "__main__":
    main()