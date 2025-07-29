import streamlit as st
import pandas as pd
import requests

OLLAMA_MODEL = "llama3"  
OLLAMA_URL = "http://localhost:11434/api/generate"

st.set_page_config(page_title="Fast Domain Detection from Metadata", layout="wide")
st.title("🚀 Fast Dataset Domain Detection (Metadata Only)")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded dataset: {uploaded_file.name} with {len(df)} rows and {len(df.columns)} columns")
        st.subheader("Preview of your dataset (first 5 rows)")
        st.dataframe(df.head())

        if st.button("🔎 Detect Domain Using Metadata Only"):
            with st.spinner("Contacting Ollama LLM..."):

                # Prepare metadata text (columns + types + row count)
                num_rows = len(df)
                columns = df.columns.tolist()
                dtypes = df.dtypes.astype(str).to_dict()

                metadata_text = f"Number of rows: {num_rows}\nColumns and types:\n"
                for col in columns:
                    metadata_text += f" - {col}: {dtypes[col]}\n"

                # Prompt only metadata
                prompt = f"""
You are a data domain expert.
Based on the following metadata of a dataset, determine the most appropriate domain or industry it belongs to.

Metadata:
{metadata_text}

Return only the most likely domain or industry name (e.g., healthcare, finance, retail, education, social media, logistics).
"""

                # Call Ollama API with shorter timeout
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60  # 1-minute timeout for faster response
                )

                if response.status_code == 200:
                    domain = response.json().get("response", "").strip()
                    st.success(f"🧠 Detected Domain: **{domain}**")
                else:
                    st.error(f"❌ Ollama error: {response.text}")

    except Exception as e:
        st.error(f"Failed to load or process the CSV: {e}")
