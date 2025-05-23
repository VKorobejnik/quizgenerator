import os

API_KEY = os.getenv("DEEPSEEK_API_KEY") #os.getenv("OPENAI_API_KEY") 
BASE_URL = "https://api.deepseek.com"

# Model configurations
MULTILINGUAL_EMBEDDING_MODEL = "distiluse-base-multilingual-cased"
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL= "deepseek-chat"  #"gpt-4o" 

# Chunking parameters
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
DESIRED_TOPICS = 10
MIN_CHUNKS = DESIRED_TOPICS + 5
MAX_CHUNK_SIZE = 8000
MIN_CHUNK_SIZE = 500

# Language-specific configurations
LANGUAGE_CONFIG = {
                "en": {
                    "quiz_title": "Knowledge Assessment",
                    "difficulty_levels": ["Easy", "Medium", "Hard"],
                    "instructions": {
                        "create_different": "Create completely different questions than previous runs",
                        "options": "Include 1 correct and 3 incorrect answers per question",
                        "randomize": "Randomize correct answer positions (A-D)",
                        "difficulty": "For each question, indicate its difficulty level in the explanation",
                        "avoid": "Avoid starting with predictable questions",
                        "plausible": "Ensure incorrect answers are plausible but clearly wrong",
                        "explanations": "Make explanations thorough and educational",
                        "uniqueness": "Ensure each question is unique and not repeated in any form",
                        "fallback": "If needed, generate general questions about the broad topic",
                    },
                    "output_format": {
                        "question": "question",
                        "options": "options",
                        "correct": "correct_answer",
                        "explanation": "explanation"
                    }
                },
                "de": {
                    "quiz_title": "Wissensbewertung",
                    "difficulty_levels": ["Einfach", "Mittel", "Schwer"],
                    "instructions": {
                        "create_different": "Erstelle völlig unterschiedliche Fragen als bei früheren Durchläufen",
                        "options": "Füge pro Frage 1 richtige und 3 falsche Antworten hinzu",
                        "randomize": "Zufällige Positionen richtiger Antworten (A-D)",
                        "difficulty": "Indiziere für jede Frage das Schwierigkeitsniveau in der Erklärung",
                        "avoid": "Vermeide es, mit vorhersagbaren Fragen zu beginnen",
                        "plausible": "Stelle sicher, dass falsche Antworten plausibel, aber eindeutig falsch sind",
                        "explanations": "Mache Erklärungen gründlich und lehrreich",
                        "uniqueness": "Stelle sicher, dass jede Frage einzigartig ist und nicht in irgendeiner Form wiederholt wird",
                        "fallback": "Generiere bei Bedarf allgemeine Fragen zum breiten Thema"
                    },
                    "output_format": {
                        "question": "Frage",
                        "options": "Optionen",
                        "correct": "richtige_Antwort",
                        "explanation": "Erklärung"
                    }
                },
                "pl": {
                    "quiz_title": "Test Wiedzy",
                    "difficulty_levels": ["Łatwy", "Średni", "Trudny"],
                    "instructions": {
                        "create_different": "Stwórz zupełnie inne pytania niż w poprzednich wersjach",
                        "options": "Dołącz 1 poprawną i 3 niepoprawne odpowiedzi na pytanie",
                        "randomize": "Losowa pozycja poprawnej odpowiedzi (A-D)",
                        "difficulty": "Dla każdego pytania wskaż poziom trudności w wyjaśnieniu",
                        "avoid": "Unikaj przewidywalnych pytań na początku",
                        "plausible": "Niepoprawne odpowiedzi powinny być prawdopodobne, ale wyraźnie błędne",
                        "explanations": "Wyjaśnienia powinny być szczegółowe i edukacyjne",
                        "uniqueness": "Upewnijcie się, że każde pytanie jest unikalne i nie powtarza się w żadnej formie",
                        "fallback": "Jeśli to konieczne, wygeneruj ogólne pytania dotyczące szerokiego tematu",
                    },
                    "output_format": {
                        "question": "question",
                        "options": "options",
                        "correct": "correct_answer",
                        "explanation": "explanation"
                    }
                },
                "ro": {
                    "quiz_title": "Evaluare Cunoaștere",
                    "difficulty_levels": ["Ușor", "Mediu", "Dificil"],
                    "instructions": {
                        "create_different": "Creați întrebări complet diferite de rulările anterioare",
                        "options": "Includeți 1 răspuns corect și 3 incorecte pentru fiecare întrebare",
                        "randomize": "Poziția aleatorie a răspunsului corect (A-D)",
                        "difficulty": "Pentru fiecare întrebare, indicați nivelul de dificultate în explicație",
                        "avoid": "Evitați întrebări previzibile la început",
                        "plausible": "Asigurați-vă că răspunsurile incorecte sunt plauzibile dar clar greșite",
                        "explanations": "Faceți explicațiile detaliate și educative",
                        "uniqueness": "Asigurați-vă că fiecare întrebare este unică și nu se repetă în nicio formă",
                        "fallback": "Dacă este necesar, generați întrebări generale despre subiectul larg",
                    },
                    "output_format": {
                        "question": "question",
                        "options": "options",
                        "correct": "correct_answer",
                        "explanation": "explanation"
                    }
                },
                "bg": {
                    "quiz_title": "Тест за Знания",
                    "difficulty_levels": ["Лесен", "Среден", "Труден"],
                    "instructions": {
                        "create_different": "Създайте напълно различни въпроси от предишните тестове",
                        "options": "Включете 1 верен и 3 грешни отговора на въпрос",
                        "randomize": "Случайно разпределение на правилните отговори (A-D)",
                        "difficulty": "За всеки въпрос посочете нивото на трудност в обяснението",
                        "avoid": "Избягвайте предвидими въпроси в началото",
                        "plausible": "Грешните отговори трябва да са правдоподобни, но ясно грешни",
                        "explanations": "Обясненията трябва да са подробни и образователни",
                        "uniqueness": "Уверете се, че всеки въпрос е уникален и не се повтаря под никаква форма",
                        "fallback": "Ако е необходимо, генерирайте общи въпроси относно широката тема",
                    },
                    "output_format": {
                        "question": "question",
                        "options": "options",
                        "correct": "correct_answer",
                        "explanation": "explanation"
                    }
                }
            }

SYSTEM_MESSAGES= {
            "en": "You are an expert quiz designer who creates pedagogically effective assessments.",
            "de": "Du bist ein Experte im Quizdesign und erstellst pädagogisch effektive Bewertungen.",
            "pl": "Jesteś ekspertem od tworzenia skutecznych dydaktycznie testów i quizów.",
            "ro": "Ești un expert în crearea de evaluări pedagogice eficiente.",
            "bg": "Ти си експерт в създаването на педагогически ефективни викторини и куизове."
    }

TOPIC_FOCUS_KEY = {
                "en": "Focus specifically on these key topics:",
                "de": "Konzentriere dich speziell auf diese Schlüsselthemen:",
                "pl": "Skup się szczególnie na tych tematach:",
                "ro": "Concentrați-vă în special pe aceste subiecte cheie:",
                "bg": "Концентрирай се специално върху тези ключови теми.:"
    }

# Supported languages
SUPPORTED_LANGUAGES = {
    "English": "en",
    "German": "de",
    "Polski": "pl",
    "Romanian": "ro",
    "Български": "bg"
}

LANGUAGE_CODES = {code: name for name, code in SUPPORTED_LANGUAGES.items()}

#prompt templates
MAIN_QUIZ_PROCESSING_PROMPT_TEMPLATE = """
                    Generate {remaining_questions} unique quiz questions in {language_name} with these rules:
                    - RANDOMIZE correct answer positions (approximately equal distribution of A/B/C/D)
                    - Correct answers should NOT follow patterns (e.g., not all B's)
                    - When possible, distribute as:
                    - 25% A correct
                    - 25% B correct
                    - 25% C correct
                    - 25% D correct 
                    from this content chunk:
                    {chunk}
                    Random seed: {random_seed}, Timestamp: {timestamp}
                    Instructions:
                    - {create_different}
                    - {options}
                    - {randomize}
                    - Difficulty distribution:
                    - {difficulty_prompt}
                    {topic_focus_prompt}
                    - {difficulty}
                    - {avoid}
                    - {plausible}
                    - {explanations}
                    - {uniqueness}

                    Important: 
                    - Before generating each question, check that it hasn't been asked before in any form
                    - Questions should test different knowledge aspects even if they cover similar topics
                    - If you can't generate enough questions from this specific content, indicate that

                    Return JSON format:
                    {{
                        "questions": [
                            {{
                                "question": "text",
                                "options": {{
                                    "A": "option 1",
                                    "B": "option 2", 
                                    "C": "option 3",
                                    "D": "option 4"
                                }},
                                "correct_answer": "A-D",
                                "explanation": "text",
                                "difficulty": "Easy/Medium/Hard"
                            }}
                        ],
                        "content_adequate": true/false (whether this content could yield more questions)
                    }}
                    """

SECONDARY_QUIZ_PROCESSING_PROMPT_TEMPLATE = """
            Generate {remaining} additional unique quiz questions in {language_name} about these topics:
            {focus_topics}
            If topics are not present concentrate on the general concepts.

            Instructions:
            - {create_different}
            - {options}
            - {randomize}
            - Difficulty distribution:
            - {difficulty_prompt}
            - {difficulty}
            - {avoid}
            - {plausible}
            - {explanations}
            - {uniqueness}
            - {fallback}

            Important:
            - These should be general questions about the topic area
            - Ensure they don't duplicate any existing questions
            - Maintain the requested difficulty distribution

            Existing questions (do not duplicate these):
            {existing_questions}

            Return JSON format same as before.
            """