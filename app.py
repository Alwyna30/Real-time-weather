import streamlit as st
from pipeline import graph  # Import the LangGraph pipeline

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="Real-Time Weather & Resume Q&A", layout="centered")

st.title("🌤 Real-Time Weather & Resume Q&A")
st.write(
    """
    Ask a question about:
    - **Weather / Temperature** in any city  
    - **Resume details** (from the uploaded PDF)
    
    This app uses a free simulated AI response for demonstration purposes.
    """
)

# -------------------------------
# User Input Section
# -------------------------------
query = st.text_input("Enter your question:")

if st.button("Submit") and query:
    # Initialize the state for LangGraph
    state = {
        "query": query,
        "next_step": "",
        "answer": "",
        "final_answer": ""
    }

    # Process the query
    with st.spinner("Processing your request..."):
        try:
            result = graph.invoke(state)
            st.subheader("AI Response")
            st.write(result["final_answer"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.caption("Built with LangGraph + Streamlit | Free Demo without real LLMs")
