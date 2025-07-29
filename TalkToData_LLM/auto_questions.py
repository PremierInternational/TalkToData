#author : Sanjay
#version : v1.0
#date : June-2025


import streamlit as st
import pandas as pd
import requests
import json
import re

st.set_page_config(page_title="📊 AI Dataset Explorer", layout="wide")
st.title("🧠 AI Question Generator & Answerer (via Ollama REST API)")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def call_ollama(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return None
    except Exception as e:
        return None

uploaded_file = st.file_uploader("Upload Your Dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(5))

if 'df' in st.session_state:
    df = st.session_state['df']
    if st.button("🧠 Generate Questions Based on Dataset"):
        prompt = f"""
        You are a helpful AI assistant.

        Based on this dataset:
        Columns: {list(df.columns)}
        Sample data: {df.head(3).to_dict()}

        Generate 5 clear and concise questions a user might ask about this dataset.
        Respond only with a JSON list like:
        ["Question 1", "Question 2", ...]
        """

        raw_output = call_ollama(prompt)

        match = re.search(r"\[.*\]", raw_output, re.DOTALL) if raw_output else None

        if match:
            try:
                questions = json.loads(match.group())
                if isinstance(questions, list):
                    st.session_state['questions'] = questions
                    st.success("Questions generated successfully!")
                else:
                    st.error("Parsed content is not a list of questions.")
            except Exception as e:
                st.error("❌ Failed to parse JSON questions. Full response:")
                st.text(raw_output)
        else:
            st.error("❌ Could not detect valid JSON question format in model output.")
            st.text(raw_output or "No response from Ollama.")

if 'questions' in st.session_state and 'df' in st.session_state:
    st.subheader("📌 Suggested Questions")
    selected_question = st.selectbox("Choose a question to get the answer:", st.session_state['questions'])

    if st.button("💬 Get Answer"):
        df = st.session_state['df']
        answer_prompt = f"""
        You are a helpful AI assistant. Use the dataset info below to answer the question.

        Columns: {list(df.columns)}
        Sample data: {df.head(10).to_dict()}

        Question: {selected_question}

        Provide a short answer based on the dataset.
        """

        answer = call_ollama(answer_prompt)
        if answer:
            st.success(answer.strip())
        else:
            st.error("Failed to get a valid answer from Ollama.")
