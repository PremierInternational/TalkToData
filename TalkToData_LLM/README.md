# 🧠 Talk To Your Data – Streamlit App

## 📌 Description

A local, privacy-first Streamlit app that lets you upload a CSV file and ask questions in natural language. Optionally integrates with a local LLM (Ollama) for enhanced query understanding.

---

## 🚀 Features

- Upload and preview CSV data
- Ask natural-language questions about your data
- Optional integration with Ollama LLM (`http://localhost:11434`)
- Built-in visualizations: bar, line, and pie charts
- Filter and query your data with dropdowns
- No data leaves your machine (privacy focused)

---

## 📁 Project Structure

talk-to-your-data/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── logo.png # (Optional) Logo image for branding
└── README.md # This documentation file


✅ 3. Install dependencies
pip install -r requirements.txt

▶️ Running the App
python -m streamlit run app.py 

-------------------------------------------------------------------------------------------------------------------
🤖 Using Ollama LLM
To use AI-based query understanding, install Ollama and run a local model.

1. Install Ollama
Download from: https://ollama.com/download

2. Pull and Run a Model
ollama run tinyllama
Or use another supported model like mistral, llama3, or phi3.

