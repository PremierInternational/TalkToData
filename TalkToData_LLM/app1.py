#author : Sanjay
#version : v1.0
#date : June-2025

import streamlit as st
import pandas as pd
import spacy
import re
import requests
import time          # measure Ollama response time
from difflib import get_close_matches

# ──────────────────────────────────────────
#  Initial setup
# ──────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="NLP Chat + Ollama", layout="wide")
st.title("🤖 Ask Questions About Your Data (Chat + Ollama)")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

# ──────────────────────────────────────────
#  Helper: call Ollama
# ──────────────────────────────────────────
def query_ollama(prompt, model="phi"):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"]
    except Exception as e:
        return f"Error: {e}"

# ──────────────────────────────────────────
#  NLP / parsing helpers for local Q&A
# ──────────────────────────────────────────
def find_best_matching_column(text, columns):
    tokens = re.findall(r'\b\w+\b', text.lower())
    for token in tokens:
        match = get_close_matches(token, [c.lower() for c in columns], n=1, cutoff=0.8)
        if match:
            return next(c for c in columns if c.lower() == match[0])
    return None

def extract_columns_list(text, df):
    return [col for col in df.columns if col.lower() in text.lower()]

def extract_condition(question, columns):
    mapping = {
        "more than": ">", "greater than": ">",
        "less than": "<", "lower than": "<",
        "at least": ">=", "at most": "<=",
        "equals": "==", "equal to": "==",
        "is": "==", "not": "!=", "=": "=="
    }
    for phrase, op in mapping.items():
        if phrase in question:
            left, _, right = question.partition(phrase)
            col = find_best_matching_column(left, columns)
            val = right.strip()
            return col, op, val
    return None, None, None

def is_numeric(v):
    try:
        float(v)
        return True
    except:
        return False

def process_question(question, df):
    q = question.lower()
    col, op, val = extract_condition(q, df.columns)
    if col and op and val:
        try:
            query_str = f"`{col}` {op} {float(val) if is_numeric(val) else repr(val.strip())}"
            out = df.query(query_str)
            return out if not out.empty else "No matching rows found."
        except Exception as e:
            return f"Error while filtering rows: {e}"

    # simple FAQs
    if "how many rows" in q:           return f"Total rows: {df.shape[0]}"
    if "how many columns" in q:        return f"Total columns: {df.shape[1]}"
    if "column names" in q:            return ", ".join(df.columns)
    if "first" in q and "row" in q:    return df.head(5)
    if "last" in q and "row" in q:     return df.tail(5)
    if "sample" in q or "random" in q: return df.sample(5)

    if "null" in q or "missing" in q:
        col = find_best_matching_column(q, df.columns)
        return f"Missing in {col}: {df[col].isnull().sum()}" if col else "Specify a column"

    if "unique" in q:
        col = find_best_matching_column(q, df.columns)
        return df[col].unique() if col else "Specify a column"

    if "max" in q:
        col = find_best_matching_column(q, df.columns)
        return f"Max of {col}: {df[col].max()}" if col else "Specify a column"

    if "min" in q:
        col = find_best_matching_column(q, df.columns)
        return f"Min of {col}: {df[col].min()}" if col else "Specify a column"

    if "average" in q or "mean" in q:
        cols = extract_columns_list(q, df)
        return {c: df[c].mean() for c in cols} if cols else "No numeric columns found."

    if "duplicate" in q:               return df[df.duplicated()]

    return "Sorry, I didn't understand. Try asking about rows, columns, filtering, etc."

# ──────────────────────────────────────────
#  Main app
# ──────────────────────────────────────────
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 💬 Ask Your Question")
    user_question = st.text_input("Type your question…", key="q")
    use_ollama = st.checkbox("🤖 Use Ollama AI")

    if user_question:
        with st.spinner("Thinking…"):
            # ─── Using Ollama ────────────────────────────
            if use_ollama:
                prompt = (
                    "You are a data expert.\n"
                    f"Columns in the DataFrame: {list(df.columns)}.\n"
                    f"Answer the following question: \"{user_question}\""
                )

                t0 = time.time()
                ollama_text = query_ollama(prompt)
                t1 = time.time()
                st.metric("🕒 Ollama Response Time (sec)", round(t1 - t0, 2))

                st.markdown("### 🤖 Ollama Response")
                st.write(ollama_text)       # show raw text only

                # Optional: try executing returned code silently
                try:
                    local = {"df": df}
                    exec(f"result = {ollama_text}", {}, local)
                    _ = local["result"]     # not displayed to avoid UI errors
                except Exception:
                    pass

            # ─── Local rule‑based Q&A ────────────────────
            else:
                result = process_question(user_question, df)

                st.markdown("### ✅ Answer")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, dict):
                    st.json(result)
                else:
                    st.write(result)
else:
    st.info("👈 Upload a CSV to get started.")
