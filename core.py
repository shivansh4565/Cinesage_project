from dotenv import load_dotenv
from typing import List, Optional

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")


class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

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

movie_text = input("Enter movie description:\n")

chain = prompt | model | parser

response = chain.invoke(
    {
        "paragraph": movie_text,
        "format_instructions": parser.get_format_instructions(),
    }
)

print(response)