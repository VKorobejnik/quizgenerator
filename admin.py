import streamlit as st
import json
import base64
import bcrypt

USER_DATA_FILE = "user_data.b64"

# Shared helpers
def load_user_data():
    try:
        with open(USER_DATA_FILE, "r") as f:
            encoded = f.read()
            decoded = base64.b64decode(encoded.encode()).decode()
            return json.loads(decoded)
    except:
        return {}

def save_user_data(data):
    encoded = base64.b64encode(json.dumps(data).encode()).decode()
    with open(USER_DATA_FILE, "w") as f:
        f.write(encoded)

def hash_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

# Admin UI
st.title("Admin Panel 👨‍💻")
st.write("Manage authorized users for the quiz app.")

# Password-protect admin access (optional)
admin_password = st.text_input("Admin Password", type="password")
#if admin_password != os.getenv("ADMIN_PASSWORD"):  # Set env var for security
    #st.error("Unauthorized")
    #st.stop()

# Add new user
st.subheader("Add User")
new_username = st.text_input("New Username")
new_password = st.text_input("New Password", type="password")
if st.button("Add User"):
    user_data = load_user_data()
    if new_username in user_data:
        st.error("User already exists!")
    else:
        user_data[new_username] = hash_password(new_password)
        save_user_data(user_data)
        st.success(f"Added user: {new_username}")

# List/delete users
st.subheader("Current Users")
user_data = load_user_data()
for user in user_data:
    cols = st.columns([3, 1])
    cols[0].write(user)
    if cols[1].button(f"Delete {user}"):
        user_data.pop(user)
        save_user_data(user_data)
        st.rerun()