# Real-Time Weather & Resume Q&A Agent

This project is an AI-powered agent built using LangChain, LangGraph, and LangSmith.  
It can:

- Fetch **real-time weather and temperature data** using the OpenWeatherMap API  
- Provide **human-friendly weather summaries** using LLaMA 3 (Ollama)  
- Answer **questions from a PDF document** using RAG (Retrieval-Augmented Generation)  
- Decide **automatically** whether to call the Weather API or search the PDF using LangGraph  
- Log and evaluate all responses in **LangSmith**  
- Include a **Streamlit UI** for an interactive demo

---

## Features

1. **Real-Time Weather & Temperature**  
   - Fetches the **current temperature** and weather conditions for any city  
   - Returns professional, user-friendly summaries

2. **RAG (PDF Q&A)**  
   - Answers questions from a PDF (like a resume)  
   - Uses embeddings and vector database search

3. **LangGraph Branching Logic**  
   - Automatically decides whether to trigger **Weather API** or **PDF Q&A**

4. **LangSmith Logging**  
   - Records **weather queries** and **PDF answers** for evaluation

5. **Streamlit UI**  
   - Simple web interface to interact with the AI agent

6. **Unit Testing**  
   - Validates Weather API, PDF RAG, and full pipeline

---

## Project Structure


Real-time-weather/
│
├── Weather Data.ipynb # Main notebook
├── app.py # Streamlit UI
├── requirements.txt # Dependencies
├── README.md # Project Documentation
├── Alwyna Data Science Resume.pdf # PDF for RAG


---

## Setup Instructions

1. Clone the repository
git clone https://github.com/Alwyna30/Real-time-weather.git <br>
cd Real-time-weather

2. Install dependencies
pip install -r requirements.txt

3. Run the Jupyter Notebook
jupyter notebook

Weather Data .ipynb

Run all cells to test weather/temperature responses and PDF Q&A

## Run the Streamlit App
streamlit run app.py
Then open your browser at http://localhost:8501.

Example queries:

"What is the weather in Mumbai today?" → Returns temperature & weather summary

"Tell me about your education" → Returns answer from PDF

## Screenshots
1. LangSmith Trace
(Insert screenshot showing temperature + PDF logs)

2. Weather Query Result
(Insert screenshot showing temperature output)

3. Streamlit UI
(Insert screenshot of the running Streamlit app)

## Demo Video
Demo Video Link: [Insert Google Drive or Loom link here]

## Evaluation Checklist
 Fetches real-time temperature and weather
 Answers questions from PDF (RAG)
 Branching logic using LangGraph
 LangSmith logging enabled
 Streamlit UI demo working
 README.md includes screenshots and demo link
