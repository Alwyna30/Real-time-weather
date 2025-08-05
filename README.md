# Real-Time Weather & Resume Q&A Agent

This project is an AI-powered agent built using LangChain, LangGraph, and LangSmith.  
It demonstrates:

Real-Time Weather Fetching using OpenWeatherMap API

PDF Question Answering using Retrieval-Augmented Generation (RAG)

LangGraph for building a simple multi-node AI pipeline

Streamlit UI for interactive queries
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
#### 1. LangSmith Trace

<img width="1919" height="914" alt="image" src="https://github.com/user-attachments/assets/4ab2ea1c-6a9f-4c23-87bd-7fa988db5de1" />

<img width="1913" height="910" alt="image" src="https://github.com/user-attachments/assets/2eb122ad-c167-4d1b-b47a-1e889bf19615" />


#### 2. Weather Query Result

<img width="1912" height="912" alt="image" src="https://github.com/user-attachments/assets/c526faf5-9c81-4362-9d36-dcafa6de09c7" />


#### 3. Streamlit UI
(https://real-time-weather-teq3kzsyc38vbvdzaose9l.streamlit.app/)

## Demo Video
Demo Video Link: (https://drive.google.com/file/d/1ww-nMnUg8m9dqgVXjvGHvKUONpzBtKMh/view?usp=sharing)

## Conclusion
This project successfully demonstrates how to combine real-time weather data retrieval and PDF-based question answering in a single interactive Streamlit application.

Key takeaways:

Integrated LangGraph for a modular AI workflow.

Implemented RAG (Retrieval-Augmented Generation) for resume queries.

Achieved real-time weather updates using OpenWeatherMap API.

Deployed a fully functional Streamlit web app ready for public use.
