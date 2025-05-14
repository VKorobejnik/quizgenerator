import streamlit as st
from core.utils import database_exists
from core.tabs import document_processor_tab


def show():
    
    # Initialize session state for db_exists if not already set
    if "db_exists" not in st.session_state:
        st.session_state["db_exists"] = database_exists()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Document Processor", "Topic Extractor", "Quiz Generator", "Quiz Editor", "Help"])  # Added 4-th tab

    with tab1:  
        document_processor_tab()
    
    with tab2:
        from core.tabs import topic_extractor_tab
        topic_extractor_tab()

    with tab3:
        from core.tabs import quiz_generator_tab
        quiz_generator_tab()
        
    with tab4:
        from core.tabs import quiz_editor_tab
        quiz_editor_tab()
        
    with tab5:  
        from core.tabs import help_tab
        help_tab()

