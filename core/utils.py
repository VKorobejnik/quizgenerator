import re
import os
import shutil
import streamlit as st
import json
from datetime import datetime
import gc
import ctypes
import ctypes.util
from app_config import LANGUAGE_CODES, LANGUAGE_CONFIG, TOPIC_FOCUS_KEY, SYSTEM_MESSAGES


def get_language_name(language_code="en"):
    return LANGUAGE_CODES.get(language_code)

def get_language_config(language_code="en"):
    return LANGUAGE_CONFIG.get(language_code)

def get_system_messages(language_code="en"):
    return SYSTEM_MESSAGES.get(language_code)

def get_topic_focus_key(language_code="en"):
    return TOPIC_FOCUS_KEY.get(language_code)

def get_ui_text(key, language_code="en"):
    """Helper function to get localized UI text"""
    UI_TEXT = {
                "en": {
                    "description": "Description",
                    "select": "Select"
                },
                "de": {
                    "description": "Beschreibung",
                    "select": "Wähle"
                },
                
                "pl": {
                    "description": "Opis",
                    "select": "Wybierz",
                },
                "ro": {
                    "description": "Descriere",
                    "select": "Selectați",
                },
                "bg": {
                    "description": "Описание",
                    "select": "Изберете",
                }
            }
    return UI_TEXT.get(language_code, UI_TEXT["en"]).get(key, UI_TEXT["en"][key])

def clean_document_content(text):
    """Clean document content by removing patterns"""
    replace_patterns = {
        r'©.*?\n': '',
        r'Copyright.*?\n': '',
        r'All rights reserved.*?\n': '',
        r'Version \d+\.\d+': '',
        r'Confidential.*?\n': '',
        r'Proprietary.*?\n': '',
        r'Page \d+ of \d+': '',
        r'Doc ID:.*?\n': ''
    }

    for pattern, replacement in replace_patterns.items():
        text = re.compile(pattern).sub(replacement, text)

    return text

def validate_json(json_str):
    """Validate that the JSON string is properly formatted."""
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {e}")
        return False
    
def save_json_file(file_path, content):
    """Save JSON content to a file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
        st.success(f"File saved successfully to {file_path}")
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False
    
def decode_unicode(text):
    return bytes(text, "utf-8").decode("unicode_escape")

def encode_unicode(text):
    return json.dumps(text, ensure_ascii=False)

def ensure_sample_data_dir():
    """Ensure the sample_data directory exists"""
    os.makedirs("sample_data", exist_ok=True)

def database_exists():
    """Check if FAISS database exists"""
    return os.path.exists("sample_data/faiss_db/index.faiss")

def purge_database():
    """Purge the FAISS database files with better error handling"""
    try:
        if not database_exists():
            return False, "No database found to purge"

        # Remove all files in the sample_data directory
        for filename in os.listdir("sample_data"):
            file_path = os.path.join("sample_data", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                return False, f"Failed to delete {file_path}: {e}"
        # Remove all generated topics
        for filename in os.listdir("topics_data"):
            file_path = os.path.join("topics_data", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                return False, f"Failed to delete {file_path}: {e}"

        # Update session state to reflect database deletion
        KEEP_KEYS = ['logged_in', 'username']

        # Delete all except specified keys
        for key in list(st.session_state.keys()):
            if key not in KEEP_KEYS:
                del st.session_state[key]
        cleanup_memory()        
        #st.session_state["db_exists"] = False  
        #st.session_state["uploaded_file_name"] = None  # Purge uploaded document from session_state
        #st.session_state["document_processed"] = False  # Reset document processed flag
        return True, "Database purged successfully!"
    except Exception as e:
        return False, f"Error during purge: {e}" 

def chunk_content(text, max_chars=30000, min_chunk_size=1000):
    """Split content into semantically meaningful chunks

    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (default: 30,000)
        min_chunk_size: Minimum size to consider as standalone chunk (default: 1,000)

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""

    # First try splitting by paragraphs
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    # Add remaining content if meaningful
    if current_chunk and len(current_chunk.strip()) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    elif current_chunk:  # Small remaining chunk - append to last chunk if exists
        if chunks:
            chunks[-1] = chunks[-1] + "\n\n" + current_chunk.strip()

    # Fallback for texts without paragraphs
    if not chunks:
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    return chunks

def get_quiz_file_name(language_code="en"):
    filename = f"{language_code}_quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return filename  

def cleanup_memory():
    st.cache_data.clear()
    gc.collect()
    # Manually trigger Python's memory allocator to release blocks
    try:
        libc_path = ctypes.util.find_library('c')
        if libc_path is not None:
            print(f"Found C library")
            libc = ctypes.CDLL(libc_path)
            if hasattr(libc, 'malloc_trim'):
                libc.malloc_trim(0)  # Release memory to OS (Linux/glibc)
    except Exception as e:
        print(f"Could not release memory via malloc_trim: {e}")