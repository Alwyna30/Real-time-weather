import requests
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import fitz  # PyMuPDF

# ==============================
# 1️⃣ STATE DEFINITION
# ==============================
class InterviewState(TypedDict):
    query: str          # User's input question
    next_step: str      # Which node to go next (weather_api / rag_pipeline)
    answer: str         # Raw answer (from API or PDF)
    final_answer: str   # Final human-friendly response

# ==============================
# 2️⃣ WEATHER API NODE
# ==============================
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_KEY"  # <-- Replace with your key

def weather_api_node(state: InterviewState) -> InterviewState:
    """Fetches real-time weather data using OpenWeatherMap."""
    query = state['query'].lower()
    
    # Extract city name
    if " in " in query:
        city = query.split(" in ")[-1].replace("?", "").strip()
        for word in ["today", "tomorrow", "now", "currently"]:
            city = city.replace(word, "").strip()
    else:
        city = query
    
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

# ==============================
# 3️⃣ PDF / RAG PIPELINE NODE
# ==============================
pdf_path = "Alwyna Data Science Resume.pdf"  # Ensure this PDF is in the folder

# Extract PDF text
doc = fitz.open(pdf_path)
resume_text = "".join([page.get_text() for page in doc])

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
resume_chunks = text_splitter.split_text(resume_text)

# Create in-memory Qdrant vector store
from langchain_ollama import OllamaEmbeddings  # You can remove if Streamlit fails
try:
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
except:
    embedding_model = None  # If Ollama is not available, fallback to dummy

qdrant = QdrantClient(":memory:")
vectorstore = Qdrant.from_texts(
    texts=resume_chunks,
    embedding=embedding_model if embedding_model else lambda x: [[0]*384]*len(x),
    location=":memory:",
    collection_name="resume_embeddings"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def rag_pipeline_node(state: InterviewState) -> InterviewState:
    """Fetches relevant info from PDF using RAG (vector search)."""
    query = state['query']
    try:
        relevant_docs = retriever.invoke(query)
    except:
        relevant_docs = []
    
    if not relevant_docs:
        state["answer"] = "Sorry, I couldn't find relevant information in the PDF."
    else:
        combined_context = "\n".join([doc.page_content for doc in relevant_docs])
        state["answer"] = f"Based on the PDF:\n{combined_context}"
    return state

# ==============================
# 4️⃣ SIMULATED LLM RESPONSE
# ==============================
def llm_processing_node(state: InterviewState) -> InterviewState:
    """Simulates an AI response (free) for Streamlit."""
    query = state['query']
    answer = state['answer']
    state["final_answer"] = (
        f"Professional AI Response (Simulated):\n\n"
        f"User asked: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"(Free demo without real LLM)"
    )
    return state

# ==============================
# 5️⃣ DECISION NODE
# ==============================
def user_input_node(state: InterviewState) -> InterviewState:
    return state

def decision_node(state: InterviewState) -> InterviewState:
    query = state['query'].lower()
    if "weather" in query or "temperature" in query:
        state["next_step"] = "weather_api"
    else:
        state["next_step"] = "rag_pipeline"
    return state

# ==============================
# 6️⃣ BUILD LANGGRAPH PIPELINE
# ==============================
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
        "rag_pipeline": "rag_pipeline"
    }
)
graph_builder.add_edge("weather_api", "llm_processing")
graph_builder.add_edge("rag_pipeline", "llm_processing")
graph_builder.add_edge("llm_processing", END)

# Compile final graph
graph = graph_builder.compile()
