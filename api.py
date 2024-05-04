import os
import bs4
from dotenv import load_dotenv

from fastapi import FastAPI, Body


from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



# ################################### FastAPI setup ############################################
class Settings(BaseSettings):
    openapi_url: str = ""

settings = Settings()

app = FastAPI(openapi_url=settings.openapi_url)

# origins = ["http://localhost:8081"]
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class Input(BaseModel):
    question: str
    chat_history: list

class Metadata(BaseModel):
    conversation_id: str

class Config(BaseModel):
    metadata: Metadata
    
class RequestBody(BaseModel):
    input: Input 
    config: Config

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chatbot")
async def chat(
    query: RequestBody = Body(...),
):
    print(query.input.question)
    return rag_chain.invoke(query.input.question)
    