import json
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatMistralAI(model="mistral-small-2506")

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are an expert movie information extraction system.

Your task is to carefully analyze the movie description provided below and extract all important information.

Rules:
- Use only the information explicitly mentioned in the text.
- Do not make up facts.
- If a field is not mentioned, return null.
- Return ONLY valid JSON.
- Do NOT wrap the JSON inside markdown.
- Do NOT write any explanation.
- Also generate a concise summary (2-3 sentences).

Extract the following information in this exact JSON format:

{{
    "movie_name": "",
    "release_year": "",
    "director": "",
    "genre": [],
    "cast": [],
    "plot": "",
    "setting": "",
    "main_theme": [],
    "rating": {{
        "source": "",
        "value": ""
    }},
    "awards": [],
    "music_by": "",
    "language": "",
    "country": "",
    "duration": "",
    "production_company": "",
    "keywords": [],
    "summary": ""
}}

Movie Description:
{text}
""")

chain = prompt | model

# ---------------- UI ---------------- #

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Information Extractor")
st.write("Paste a movie description below and extract structured information using AI.")

movie_text = st.text_area(
    "Movie Description",
    height=250,
    placeholder="Paste the movie description here..."
)

if st.button("Extract Information", type="primary"):

    if not movie_text.strip():
        st.warning("Please enter a movie description.")
        st.stop()

    with st.spinner("Extracting information..."):

        response = chain.invoke({"text": movie_text})

        content = response.content.strip()

        # Remove markdown if present
        if content.startswith("```"):
            content = (
                content.replace("```json", "")
                .replace("```", "")
                .strip()
            )

        try:
            data = json.loads(content)

            st.success("Information extracted successfully!")

            st.subheader("Structured JSON")
            st.json(data)

            st.subheader("Summary")
            st.write(data.get("summary", "No summary available."))

        except json.JSONDecodeError:

            st.error("The model returned invalid JSON.")

            st.subheader("Raw Response")
            st.code(content, language="text")