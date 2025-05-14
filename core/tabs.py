import streamlit as st
import os
import io
import json
import time
import shutil
from streamlit_js_eval import streamlit_js_eval
from app_config import SUPPORTED_LANGUAGES
from  core.utils import database_exists, purge_database, save_quiz_file
from core.utils import database_exists, get_ui_text, validate_json
from core.document_processor import process_document_with_semantic_preprocessing, process_document, extract_text_from_file, process_start_document, load_vector_db, query_faiss_for_quiz, generate_mcq_json



def document_processor_tab():
    #st.write(f"Running file: {os.path.basename(__file__)}")
    st.markdown("""
    This part processes a start document and embeds the document in a FAISS vector database 
    for use in the Quiz Generator. If the start document is already processed go to the Topic Extractor or to the Quiz Generator tab.
    """)

    # Initialize session state
    if 'last_action' not in st.session_state:
        st.session_state.last_action = None
        st.session_state.last_action_message = None
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "db_exists" not in st.session_state:
        st.session_state["db_exists"] = database_exists()
    if 'original_content' not in st.session_state:
        st.session_state.original_content = None
    if 'edited_content' not in st.session_state:
        st.session_state.edited_content = None
    if 'file_path' not in st.session_state:
        st.session_state.file_path = None
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    # Database status section
    st.markdown("Database Status")   
    db_path = "sample_data/faiss_db"
    metadata_path = os.path.join(db_path, "metadata.json")
    
    metadata_exists = os.path.exists(metadata_path)
    # Check metadata existence
    if not metadata_exists:
        st.warning("⚠️ It looks like no document has been processed yet. Please upload and process a document to continue.")
        #return

    # Load metadata
    if metadata_exists:
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            document_name = metadata.get("document_name")

            if not document_name:
                st.error("❌ Document name not found in metadata")
                return

            document_path = os.path.join(db_path, document_name)
            if not os.path.exists(document_path):
                st.warning(f"⚠️ Document '{document_name}' not found in database directory")
                return

        except Exception as e:
            st.error(f"❌ Error loading metadata: {str(e)}")
            return

        # Display document info panel
        with st.container(border=True):
            st.success(f"✅ Database loaded successfully")
            st.markdown(f"""
            - Document name: {document_name}
            - Processed on: {metadata.get('created_at', 'N/A')}
            """)

        # Purge button with confirmation and better error handling
        if st.button("Purge Database", help="⚠️ Warning: This will permanently delete all stored document embeddings. "
            "You'll need to re-upload documents after purging."):
            st.session_state.last_action = "purge"
            success, message = purge_database()
            st.session_state.last_action_message = message

            if success:
                st.success(message)
            else:
                st.error(message)

            # Reset document processing state
            st.session_state.document_processed = False
            st.session_state.uploaded_file_name = None

            # Force UI update
            st.rerun()

        # Display last action message if exists
        if st.session_state.last_action_message:
            if st.session_state.last_action == "purge":
                if "success" in st.session_state.last_action_message.lower():
                    st.success(st.session_state.last_action_message)
                else:
                    st.error(st.session_state.last_action_message)

    # File upload section
    st.markdown("Upload new start document if necessary. The old start document (if any) is overwritten by a new one.")
    uploaded_file = st.file_uploader(
        "Choose a Word or a PDF document", 
        type=["docx", "pdf"],
        accept_multiple_files=False,
        help="The uploaded document will be processed and embeded in the vector database"
    )

    if uploaded_file and not st.session_state.document_processed:
        loading_gif = st.image("images/Dance.gif")
        with st.spinner("Processing document..."):
            try:
                process_start_document(uploaded_file)

                # Store processed state in session
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.last_action = "upload"
                st.session_state.last_action_message = "Document processed and embeded successfully!"
                st.session_state["document_processed"] = True
                st.success(st.session_state.last_action_message)
                st.rerun()

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")  
        loading_gif.empty()
    else:
        if st.button("Logout", help="Leave the application and go to the Login page.", key="logout_document_processor"):
            logout_placeholder = st.empty()
            logout_placeholder.info("Logging out...")
            st.session_state.logged_in = False
            st.rerun()
        
def topic_extractor_tab(): 
    st.markdown("Analyze the document stored in the FAISS database and extract Key Topics. If no Key Topics are generated and not selected, the quiz questions will be randomly drawn from the Start Document.")

    db_path = "sample_data/faiss_db"
    topics_dir = "topics_data"
    os.makedirs(topics_dir, exist_ok=True)  # Ensure topics directory exists

    metadata_path = os.path.join(db_path, "metadata.json")

    db_exists = database_exists()

    # Check database existence
    if not db_exists:
        st.warning("⚠️ It looks like no document has been processed yet. Please upload and process a document to continue.")
        return

    # Check metadata existence
    if not os.path.exists(metadata_path):
        st.warning("⚠️ Metadata file not found - please reprocess your document")
        return

    # Load metadata
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        document_name = metadata.get("document_name")

        if not document_name:
            st.error("❌ Document name not found in metadata")
            return

        document_path = os.path.join(db_path, document_name)
        if not os.path.exists(document_path):
            st.warning(f"⚠️ Document '{document_name}' not found in database directory")
            return

    except Exception as e:
        st.error(f"❌ Error loading metadata: {str(e)}")
        return

    # Display document info panel
    with st.container(border=True):
        st.success(f"✅ Database loaded successfully")
        st.markdown(f"""
        - Document name: {document_name}
        - Processed on: {metadata.get('created_at', 'N/A')}
        """)


    # Check if topics.json exists for this document
    topics_file = os.path.join(topics_dir, "topics.json")
    topics_exist = os.path.exists(topics_file)

    if topics_exist:
        try:
            with open(topics_file, "r") as f:
                topics_data = json.load(f)
            
            doc_language = topics_data.get("language", "en")
            
            st.markdown("Key Topics. You may select Key Topics to be used by the Quiz Generator. If no Key Topics are selected or generated, the quiz questions will be randomly drawn from the Start Document.")
            for i, topic in enumerate(topics_data["topics"]):
                with st.expander(f"🔷 {i+1}. {topic['topic_name']}"):
                    st.markdown(f"{get_ui_text('description', doc_language)}: {topic['description']}")

                    # Initialize session state for each topic if not exists
                    if f"selected_{i}" not in st.session_state:
                        st.session_state[f"selected_{i}"] = topic['topic_name'] in st.session_state.get("selected_topics", [])

                    # Create a checkbox for each topic
                    is_selected = st.checkbox(
                        f"{get_ui_text('select', doc_language)} {topic['topic_name']}",
                        value=st.session_state[f"selected_{i}"],
                        key=f"checkbox_{i}"
                    )

                    # Update the session state for this topic
                    st.session_state[f"selected_{i}"] = is_selected

            # Update the selected_topics list based on checkbox states
            selected_topics = [
                topic['topic_name'] for i, topic in enumerate(topics_data["topics"]) 
                if st.session_state.get(f"selected_{i}", False)
            ]

            # Update the main selected_topics in session state
            st.session_state.selected_topics = selected_topics
            st.write(f"Selected {len(selected_topics)} topic(s)")
            #col1, col2, col3 = st.columns(3)
            col1, col2, col3 = st.columns([3, 9, 3])
            with col1:
                if st.button("Remove Topics", help="Remove extracted key topics", key="remove_key_topics"):
                    try:
                        if os.path.isfile(topics_file) or os.path.islink(topics_file):
                            os.unlink(topics_file)
                        elif os.path.isdir(topics_file):
                            shutil.rmtree(topics_file)
                        st.session_state["process_clicked"] = False  # <<< Prevent unintended processing
                    except Exception as e:
                        return False, f"Failed to delete {topics_file}: {e}"                     
                    st.rerun()  
            with col3:
                if st.button("Logout", help="Leave the editor and go to the Login screen", key="logout_topic_extractor"):
                    streamlit_js_eval(js_expressions="parent.window.location.reload()")
                

        except Exception as e:
            st.error(f"❌ Error loading topics data: {str(e)}")
            print(f"Error loading topics data: {str(e)}")
            topics_exist = False  
    else:        
         with open(document_path, "rb") as f:
            file_content = f.read()
            uploaded_file = io.BytesIO(file_content)
            uploaded_file.name = document_name  
            document_text = extract_text_from_file(uploaded_file)
            # Language selection
            current_language_code = st.session_state.get("document_language", "en")
            language_names = list(SUPPORTED_LANGUAGES.keys())
            default_index = language_names.index(
                next(name for name, code in SUPPORTED_LANGUAGES.items() if code == current_language_code)
            )
            with st.expander("Key Topic Language"):
                selected_language = st.radio(
                    label ="Languages",                   
                    options=language_names,
                    key="unique_topics",
                    index=default_index,
                    horizontal=True,
                    label_visibility="hidden" 
                )

            language_code = SUPPORTED_LANGUAGES[selected_language]
            with st.expander("Advanced Settings"):            
                    num_topics = st.slider(
                    "Number of Key Topics",
                    min_value=5,
                    max_value=30,
                    value=10
                    )
            col1, col2 = st.columns([1.5, 6.0])
            with col1:
                if st.button("Extract Topics", help="Analyze the document content and extract key topics using LLM"):
                    st.session_state["process_clicked"] = True
            with col2:
                use_semantic = st.checkbox(
                    "Use semantic pre-processing",
                    help="Apply advanced Sentence Transformer NLP techniques before LLM powered topic extraction. Works only with large documents, small documents are still processed directly by LLM.",
                    value=False
                )

            # This will appear below the columns when processing
            if st.session_state.get("process_clicked", False):
                    loading_gif = st.image("images/Acrobat.gif")
                    with st.spinner(f"Analyzing {document_name}..."):
                        try:
                            if use_semantic:
                                print("Using semantic pre-processing")
                                results = process_document_with_semantic_preprocessing(
                                    document_text, 
                                    document_name,
                                    max_topics = num_topics, 
                                    language_code=language_code)
                                st.rerun()
                            else:
                                results = process_document(
                                    document_text,
                                    os.path.splitext(document_name)[0],
                                    max_topics=num_topics,
                                    language_code=language_code
                                )

                            st.session_state["topics"] = results["topics"]
                            st.session_state["process_clicked"] = False  # Reset flag
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
                            print(f"Processing failed: {str(e)}")
                        finally:
                            loading_gif.empty()
                            
def quiz_generator_tab():
    # Database status section
    st.markdown("Database Status")   
    db_path = "sample_data/faiss_db"
    metadata_path = os.path.join(db_path, "metadata.json")
    
    metadata_exists = os.path.exists(metadata_path)
    # Check metadata existence
    if not metadata_exists:
        st.warning("⚠️ It looks like no document has been processed yet. Please upload and process a document to continue.")
        #return
    if metadata_exists:
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            document_name = metadata.get("document_name")

            if not document_name:
                st.error("❌ Document name not found in metadata")
                return

            document_path = os.path.join(db_path, document_name)
            if not os.path.exists(document_path):
                st.warning(f"⚠️ Document '{document_name}' not found in database directory")
                return

        except Exception as e:
            st.error(f"❌ Error loading metadata: {str(e)}")
            return
        
        # Display document info panel
        with st.container(border=True):
            st.success(f"✅ Database loaded successfully")
            st.markdown(f"""
            - Document name: {document_name}
            - Processed on: {metadata.get('created_at', 'N/A')}
            """)
        
        # Language selection
        current_language_code = st.session_state.get("document_language", "en")
        language_names = list(SUPPORTED_LANGUAGES.keys())
        default_index = language_names.index(
                next(name for name, code in SUPPORTED_LANGUAGES.items() if code == current_language_code)
            )
        with st.expander("Quiz Language"):
            selected_language = st.radio(
                label ="Languages",
                options=language_names,
                index=default_index,
                horizontal=True,
                label_visibility="hidden" 
            )

        language_code = SUPPORTED_LANGUAGES[selected_language]
        
        with st.expander("Advanced Settings"):
            
            num_questions = st.slider(
            "Number of Questions",
            min_value=5,
            max_value=30,
            value=10
            )
        
            # Difficulty distribution
            st.write("Difficulty Distribution")
            col1, col2, col3 = st.columns(3)
        
            with col1:
                easy_pct = st.slider("Easy (%)", 0, 100, 40)
            with col2:
                medium_pct = st.slider("Medium (%)", 0, 100, 30)
            with col3:
                hard_pct = st.slider("Hard (%)", 0, 100, 30)
        
            # Normalize percentages to ensure they sum to 100%
            total_pct = easy_pct + medium_pct + hard_pct
            if total_pct != 100:
                st.warning(f"Percentages sum to {total_pct}%. They will be normalized to 100%.")
                easy_pct = int(easy_pct * 100 / total_pct)
                medium_pct = int(medium_pct * 100 / total_pct)
                hard_pct = 100 - easy_pct - medium_pct
        
            # Calculate actual question counts
            easy_count = int(num_questions * easy_pct / 100)
            medium_count = int(num_questions * medium_pct / 100)
            hard_count = num_questions - easy_count - medium_count
        
            st.write(f"Distribution: {easy_count} Easy, {medium_count} Medium, {hard_count} Hard")
        
            difficulty_distribution = {
                "easy": easy_count,
                "medium": medium_count,
                "hard": hard_count
            }
            # Initialize session state if not exists
            if "generate_clicked" not in st.session_state:
                st.session_state["generate_clicked"] = False

        # Create a top-level container for the GIF (outside columns)
        

        col1, col2, col3 = st.columns([3, 9, 3])
        with col1:
                if st.button("Generate Quiz", help="Click to generate a new quiz with the specified number of questions and difficulty distribution. "
                        "The quiz will be based on the content from your uploaded document."):              
                    st.session_state["generate_clicked"] = True

        with col3:
                if st.button("Logout", help="Leave the editor and go to the Login screen", key="logout_quiz_generator"):
                    streamlit_js_eval(js_expressions="parent.window.location.reload()")

            # Processing section (appears below columns)
        if st.session_state.get("generate_clicked", False):
                # Show full-width GIF using container width
                gif_placeholder = st.empty()
                gif_placeholder.image("images/Robot.gif")
                
                with st.spinner("Generating..."):
                    try:
                        start_time = time.time()
                        vector_db = load_vector_db()
                        end_time = time.time()
                        print(f"Vector DB loaded in {end_time - start_time:.2f} seconds")
                        
                        if not vector_db:
                            st.error("Database not found. Please upload a document first.")
                            gif_placeholder.empty()
                            st.session_state["generate_clicked"] = False
                            st.rerun()
                            
                        start_time = time.time()
                        content = query_faiss_for_quiz(vector_db)
                        end_time = time.time()
                        print(f"Content retrieved from FAISS db in {end_time - start_time:.2f} seconds")
                        
                        if not content:
                            st.error("No content retrieved from database")
                            gif_placeholder.empty()
                            st.session_state["generate_clicked"] = False
                            st.rerun()
                            
                        start_time = time.time()
                        quiz_data = generate_mcq_json(content, num_questions, difficulty_distribution, language_code)
                        end_time = time.time()
                        print(f"Quiz generated in {end_time - start_time:.2f} seconds")
                        
                        if quiz_data:
                            filename = save_quiz_file(quiz_data)
                            st.success(f"Quiz saved to {filename}")
                            st.download_button(
                                label="Download Quiz JSON",
                                data=json.dumps(quiz_data, indent=2),
                                file_name=os.path.basename(filename),
                                mime="application/json"
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Quiz generation failed: {str(e)}")
                        print(f"Quiz generation failed: {str(e)}")
                        
                    finally:
                        gif_placeholder.empty()
                        st.session_state["generate_clicked"] = False

  

            
            
def quiz_editor_tab():  
    uploaded_file = st.file_uploader(
        "Upload JSON Quiz File", 
        type=['json'],
        accept_multiple_files=False,
        help="The uploaded Quiz file will be then loaded into editor"
        )

    if uploaded_file is not None:
        try:
            content = json.load(uploaded_file)
            st.session_state.original_content = content
            #decoded_content = decode_unicode(json.dumps(content, ensure_ascii=False))
            st.session_state.edited_content = json.dumps(content, indent=2,  ensure_ascii=False)
            st.session_state.file_path = uploaded_file.name
            st.session_state.uploaded_file=uploaded_file
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON file: {e}")

        # Display editor if file is loaded
        if st.session_state.edited_content is not None:
            st.subheader("Edit JSON Content")

        # Text area for editing JSON
        edited_json = st.text_area(
            "Quiz Content",
            value=st.session_state.edited_content,
            height=600,
            key="json_editor",
            help="To stop editing the JSON file clean the uploaded file using the X button"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Validate JSON", help="Check if JSON is valid"):
                if validate_json(edited_json):
                    st.success("✅ JSON is valid!")

        with col2:
           
            #if st.download_button("💾 Save", help="Save changes (replaces original file)"):
                if validate_json(edited_json):
                    st.download_button(
                        label="Save",
                        data=edited_json,
                        file_name=st.session_state.file_path,
                        mime="application/json",
                        help="Save changes (replaces original file)"
                        )


        with col3:
            if st.button("Logout", help="Leave the editor and go to the Login screen"):
                keys_to_clear = [
                "original_content", 
                "edited_content", 
                "file_path",
                "json_editor"  # This resets the text area
                ]
                for key in keys_to_clear:
                   if key in st.session_state:
                      del st.session_state[key]
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
    else:
        if st.button("Logout", help="Leave the application and go to the Login page.", key="logout_document_editor"):
            logout_placeholder = st.empty()
            logout_placeholder.info("Logging out...")
            st.session_state.logged_in = False
            st.rerun()
                    
def help_tab():
    st.markdown("""
    ###### GlobalLogic 2025 Proof of Concept: LLM-Powered Quiz Generator  

    ###### **System Requirements**  
    - **Python Version:** 3.12.8 (Ensure you are using this exact version).  
    - **Dependencies:** Install all required modules using the `requirements.txt` file.  

    ###### **Setup Instructions**  
    1. Install dependencies by running the following command in the Windows console:  
       ```bash  
       pip install -r requirements.txt  
       ```  
    2. Navigate to the generator module directory.  
    3. Launch the GUI by running:  
       ```bash  
       streamlit run poc_generator.py  
       ```  

    ###### **Application Overview**  

    **1. Document Processor**  
    - **Purpose:** Upload and process the starting document (PDF or DOCX) for quiz generation.  
    - **Functionality:**  
      - Upload a document to embed it into the FAISS vector database.  
      - Clear the currently loaded document from the database.  

    **2. Topic Extractor**  
    - **Purpose:** Analyze the Start Document and extract Key Topics. If no Key Topics are generated and not selected, the quiz questions will be randomly drawn from the Start Document.  
    - **Prerequisite:** A document must be embedded and ready for processing by the LLM.  
    - **Functionality:** Extract key topics from the start document.  

    **Key Topic Language**  
    - Select language of the generated Key Topics. Currently supported: English, Polish, Romanian and Bulgarian. Default: English  

    **Advanced Settings**  
    - **Number of Key Topics:**  
      - Range: 5 to 30 (default: 10).  

    **3. Quiz Generator**  
    - **Prerequisite:** A document must be embedded and ready for processing by the LLM.  

    **Quiz Language**  
    - Select language of the generated Quiz. Currently supported: English, Polish, Romanian and Bulgarian. Default: English  

    **Advanced Settings**  
    - **Number of Questions:**  
      - Range: 5 to 30 (default: 10).  
    - **Difficulty Distribution:**  
      - Default: 40% Easy, 30% Medium, 30% Hard.  
      - Adjust the percentages as needed.  

    **Generating the Quiz**  
    1. Configure the settings as desired.  
    2. Click **Generate Quiz** and wait for the process to complete.  
    3. Download the resulting JSON file for use in the Quiz GUI.  

    **Quiz Modes**  
    The generated quiz can be used in:  
    - **Examination Mode** (proctored assessment).  
    - **Learning Mode** (self-paced practice).  

    **4. Quiz Editor**  
    - **Purpose:** Edit the resulting JSON Quiz file and save the edited file.  
    - **Prerequisite:** The existing JSON Quiz file.  

    **5. Quiz GUI**  
    Standalone application for running the Quiz in the two modes:  
    - **Examination Mode** (proctored assessment).  
    - **Learning Mode** (self-paced practice).  

    **System requirements:** same as for the Quiz Generator  

    Install dependencies by running the following command in the Windows console:  
    ```bash  
    pip install -r requirements.txt  
    ```  

    Navigate to the Quiz GUI directory.  
    Launch the GUI by running:  
    ```bash  
    streamlit run app.py  
    ```  
    """)