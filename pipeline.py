import requests
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
import fitz  # PyMuPDF for PDF processing

# --------------- CONFIG ----------------
OPENWEATHER_API_KEY = "f15cf1878bbc75946d59bea5d6dc0da1" 

# Initialize embedding model (free, HuggingFace)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --------------- STATE ----------------
class InterviewState(TypedDict):
    query: str
    next_step: str
    answer: str
    final_answer: str

# --------------- FUNCTIONS ----------------
def user_input_node(state: InterviewState) -> InterviewState:
    return state

def decision_node(state: InterviewState) -> InterviewState:
    """Decide if query is about weather or PDF"""
    query = state['query'].lower()
    if "weather" in query or "temperature" in query:
        state["next_step"] = "weather_api"
    else:
        state["next_step"] = "rag_pipeline"
    return state

def weather_api_node(state: InterviewState) -> InterviewState:
    """Fetch real-time weather data using OpenWeatherMap API"""
    query = state['query'].lower()

    # Clean the query to extract city name
    city = query
    for word in ["what", "is", "the", "temperature", "weather", "of", "in", "today", "now", "currently", "?", ".", ","]:
        city = city.replace(word, "")
    city = city.strip()

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url).json()
        if response.get("cod") != 200:
            state["answer"] = f"Sorry, I could not fetch the weather for {city}. Error: {response.get('message')}"
        else:
            temp = response["main"]["temp"]
            condition = response["weather"][0]["description"].capitalize()
            state["answer"] = f"The weather in {city.capitalize()} is {temp}°C with {condition}."
    except Exception as e:
        state["answer"] = f"Error fetching weather data: {str(e)}"
    return state

# --------- PDF / RAG Pipeline ---------
pdf_path = "Alwyna Data Science Resume.pdf"  # Make sure this file is in your folder
doc = fitz.open(pdf_path)

resume_text = ""
for page in doc:
    resume_text += page.get_text()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
resume_chunks = text_splitter.split_text(resume_text)

# Create in-memory Qdrant vector store
qdrant = QdrantClient(":memory:")
vectorstore = Qdrant.from_texts(
    texts=resume_chunks,
    embedding=embedding_model,
    location=":memory:",
    collection_name="resume_embeddings"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def rag_pipeline_node(state: InterviewState) -> InterviewState:
    """Answer questions from the PDF using RAG"""
    query = state['query']
    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        state["answer"] = "Sorry, I couldn't find relevant information in the PDF."
    else:
        combined_context = "\n".join([doc.page_content for doc in relevant_docs])
        state["answer"] = f"Based on the PDF:\n{combined_context}"
    return state

def llm_processing_node(state: InterviewState) -> InterviewState:
    """Simulated AI response (free, no OpenAI charges)"""
    state["final_answer"] = f"User asked: {state['query']}\n\nAnswer: {state['answer']}"
    return state

# --------------- BUILD THE GRAPH ----------------
graph_builder = StateGraph(InterviewState)
graph_builder.add_node("user_input", user_input_node)
graph_builder.add_node("decision", decision_node)
graph_builder.add_node("weather_api", weather_api_node)
graph_builder.add_node("rag_pipeline", rag_pipeline_node)
graph_builder.add_node("llm_processing", llm_processing_node)

graph_builder.add_edge(START, "user_input")
graph_builder.add_edge("user_input", "decision")
graph_builder.add_conditional_edges(
    "decision",
    lambda state: state["next_step"],
    {
        "weather_api": "weather_api",
        "rag_pipeline": "rag_pipeline",
    }
)
graph_builder.add_edge("weather_api", "llm_processing")
graph_builder.add_edge("rag_pipeline", "llm_processing")
graph_builder.add_edge("llm_processing", END)

graph = graph_builder.compile()
