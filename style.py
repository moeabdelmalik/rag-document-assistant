"""CSS styling for the RAG Assistant Streamlit app, kept out of main.py."""

CUSTOM_CSS = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        .main {
            background-color: #0f0f0f;
        }

        .stApp {
            background-color: #0f0f0f;
            color: #f0f0f0;
        }

        h1, h2, h3 {
            font-family: 'Syne', sans-serif !important;
        }

        .title-block {
            text-align: center;
            padding: 2rem 0 1rem 0;
        }

        .title-block h1 {
            font-size: 2.8rem;
            font-weight: 800;
            color: #f0f0f0;
            letter-spacing: -1px;
            margin-bottom: 0.3rem;
        }

        .title-block p {
            color: #888;
            font-size: 1rem;
            font-weight: 300;
        }

        .accent {
            color: #00e5a0;
        }

        .status-box {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-left: 3px solid #00e5a0;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            color: #ccc;
        }

        .answer-box {
            background: #151515;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            margin-top: 1.2rem;
            color: #f0f0f0;
            font-size: 0.95rem;
            line-height: 1.7;
        }

        .answer-label {
            font-family: 'Syne', sans-serif;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #00e5a0;
            margin-bottom: 0.6rem;
        }

        .chat-history {
            margin-top: 2rem;
        }

        .chat-item {
            background: #1a1a1a;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            border: 1px solid #222;
        }

        .chat-question {
            font-size: 0.8rem;
            color: #888;
            margin-bottom: 0.3rem;
            font-weight: 500;
        }

        .chat-answer {
            color: #f0f0f0;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .divider {
            border: none;
            border-top: 1px solid #222;
            margin: 2rem 0;
        }

        /* Streamlit widget overrides */
        .stFileUploader > div {
            background: #1a1a1a !important;
            border: 1px dashed #333 !important;
            border-radius: 10px !important;
        }

        .stTextInput > div > div > input {
            background: #1a1a1a !important;
            border: 1px solid #333 !important;
            color: #f0f0f0 !important;
            border-radius: 8px !important;
        }

        .stButton > button {
            background: #00e5a0 !important;
            color: #0f0f0f !important;
            font-family: 'Syne', sans-serif !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 2rem !important;
            width: 100% !important;
            transition: opacity 0.2s !important;
        }

        .stButton > button:hover {
            opacity: 0.85 !important;
        }

        .stSpinner > div {
            border-top-color: #00e5a0 !important;
        }
    </style>
"""