# LLM-Powered Quiz Generator (2025 Proof of Concept)  

**Description**:  
This LLM-powered application generates customizable quizzes in JSON format from any input document (PDF or DOCX). It extracts key topics, supports multiple languages, and allows fine-tuning of question difficulty and quantity. The generated quizzes can be used in *Examination Mode* (proctored assessments) or *Learning Mode* (self-paced practice).

---

## Application Overview  

### 1. Login  
- **Purpose:** Secure access control to the Quiz Generator application.  
- **Functionality:**  
  - Authenticate users before allowing access to features.  
  - Prevent unauthorized use by requiring login credentials.  
  - Display access denied message for unauthorized attempts.  

### 2. Document Processor  
- **Purpose:** Upload and process the starting document (PDF or DOCX) for quiz generation.  
- **Functionality:**  
  - Upload a document to embed it into the FAISS vector database.  
  - Clear the currently loaded document from the database.  

### 3. Topic Extractor  
- **Purpose:** Analyze the Start Document and extract Key Topics. If no Key Topics are generated and not selected, the quiz questions will be randomly drawn from the Start Document.  
- **Prerequisite:** A document must be embedded and ready for processing by the LLM.  
- **Functionality:** Extract key topics from the start document.  

#### Key Topic Language  
- Select language of the generated Key Topics. Currently supported: English, Polish, Romanian and Bulgarian. Default: English  

#### Advanced Settings  
- **Number of Key Topics:**  
  - Range: 5 to 30 (default: 10).  

### 4. Quiz Generator  
- **Prerequisite:** A document must be embedded and ready for processing by the LLM.  

#### Quiz Language  
- Select language of the generated Quiz. Currently supported: English, Polish, Romanian and Bulgarian. Default: English  

#### Advanced Settings  
- **Number of Questions:**  
  - Range: 5 to 30 (default: 10).  
- **Difficulty Distribution:**  
  - Default: 40% Easy, 30% Medium, 30% Hard.  
  - Adjust the percentages as needed.  

#### Generating the Quiz  
1. Configure the settings as desired.  
2. Click **Generate Quiz** and wait for the process to complete.  
3. Download the resulting JSON file for use in the Quiz GUI.  

#### Quiz Modes  
The generated quiz can be used in:  
- **Examination Mode** (proctored assessment).  
- **Learning Mode** (self-paced practice).  

### 5. Quiz Editor  
- **Purpose:** Edit the resulting JSON Quiz file and save the edited file.  
- **Prerequisite:** The existing JSON Quiz file.  

### 6. Quiz GUI (not included)  
Standalone application for running the Quiz in the two modes:  
- **Examination Mode** (proctored assessment).  
- **Learning Mode** (self-paced practice).
