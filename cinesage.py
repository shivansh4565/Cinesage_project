import streamlit as st
from dotenv import load_dotenv
from typing import List, Optional

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()

# Initialize model
model = ChatMistralAI(model="mistral-small-2506")


# Pydantic Model
class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


# Parser
parser = PydanticOutputParser(pydantic_object=Movie)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert movie information extractor.

Extract movie information from the paragraph.

{format_instructions}
""",
        ),
        ("human", "{paragraph}"),
    ]
)

# Chain
chain = prompt | model | parser

# ------------------ UI ------------------ #

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Information Extractor")
st.write("Paste a movie description below and let AI extract structured information.")

movie_text = st.text_area(
    "Movie Description",
    height=250,
    placeholder="Paste a movie description here..."
)

if st.button("Extract Information", type="primary"):

    if not movie_text.strip():
        st.warning("Please enter a movie description.")
        st.stop()

    with st.spinner("Extracting information..."):

        try:
            result = chain.invoke(
                {
                    "paragraph": movie_text,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            st.success("Extraction Complete!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Movie Details")
                st.write(f"**Title:** {result.title}")
                st.write(f"**Release Year:** {result.release_year}")
                st.write(f"**Director:** {result.director}")
                st.write(f"**Rating:** {result.rating}")

            with col2:
                st.subheader("Genres")
                st.write(result.genre)

                st.subheader("Cast")
                st.write(result.cast)

            st.subheader("Summary")
            st.info(result.summary)

            st.subheader("JSON Output")
            st.json(result.model_dump())

        except Exception as e:
            st.error("Extraction failed.")
            st.exception(e)