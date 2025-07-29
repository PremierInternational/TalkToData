#author : Sanjay
#version : v1.0
#date : June-2025


import streamlit as st
import pandas as pd
import spacy
import re
import requests
from PIL import Image
from difflib import get_close_matches

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Talk To Your Data", layout="wide")
try:
    logo = Image.open("logo.png") 
    st.image(logo, width=500)
except FileNotFoundError:
    st.warning("Logo file not found. Please check the filename and path.")
#st.image(logo, width=120)
st.title("🧠 Talk To Your Data")
st.markdown("#### 🔒 Local LLM – Privacy & Security with Your Data")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

import requests

def query_ollama(prompt, model="tinyllama"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json().get("response", "No response field in output")
    except Exception as e:
        return f"Error: {e}"


def find_best_matching_column(text, columns):
    tokens = re.findall(r'\b\w+\b', text.lower())
    for token in tokens:
        match = get_close_matches(token, [col.lower() for col in columns], n=1, cutoff=0.8)
        if match:
            return [col for col in columns if col.lower() == match[0]][0]
    return None

def extract_columns_list(text, df):
    return [col for col in df.columns if col.lower() in text.lower()]

def extract_condition(question, columns):
    condition_map = {
        "more than": ">", "greater than": ">",
        "less than": "<", "lower than": "<",
        "at least": ">=", "at most": "<=",
        "equals": "==", "equal to": "==",
        "is": "==", "not": "!=", "=": "=="
    }
    for phrase, op in condition_map.items():
        if phrase in question:
            parts = question.split(phrase)
            if len(parts) == 2:
                col = find_best_matching_column(parts[0], columns)
                val = parts[1].strip()
                return col, op, val
    return None, None, None

def is_numeric(val):
    try:
        float(val)
        return True
    except:
        return False

def process_question(question, df):
    q = question.lower()
    col, op, val = extract_condition(q, df.columns)
    if col and op and val:
        try:
            if is_numeric(val):
                val_converted = float(val)
                query_str = f"`{col}` {op} {val_converted}"
            else:
                val_converted = val.strip("'\"")
                query_str = f"`{col}` {op} @val_converted"
            result = df.query(query_str)
            return result if not result.empty else "No matching rows found."
        except Exception as e:
            return f"Error while filtering rows: {e}"

    if "how many rows" in q: return f"Total rows: {df.shape[0]}"
    if "how many columns" in q: return f"Total columns: {df.shape[1]}"
    if "column names" in q or "columns" in q: return f"Columns: {', '.join(df.columns)}"
    if "first" in q and "row" in q: return df.head(5)
    if "last" in q and "row" in q: return df.tail(5)
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
        return {col: df[col].mean() for col in cols} if cols else "No numeric columns found."
    if "duplicate" in q: return df[df.duplicated()]

    return "Sorry, I didn't understand. Try asking about rows, columns, filtering, etc."

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 💬 Ask Your Question")
    user_question = st.text_input("Type your question...", key="q")
    use_ollama = st.checkbox("🤖 Use Ollama AI")

    if user_question:
        with st.spinner("Thinking..."):
            if use_ollama:
                prompt = f"""You are a data expert.
The columns in the DataFrame are: {list(df.columns)}.
Return the answer question: "{user_question}"
"""
                response = query_ollama(prompt)
                st.markdown(response)
                try:
                    local_vars = {"df": df}
                    exec(f"result = {response}", {}, local_vars)
                    result = local_vars["result"]
                except Exception as e:
                    result = f"⚠️ Error running code: {e}"
            else:
                result = process_question(user_question, df)

        st.markdown("### ✅ Answer:")
        if isinstance(result, pd.DataFrame):
            st.dataframe(result)
        elif isinstance(result, dict):
            st.json(result)
        else:
            st.write(result)

    st.markdown("### 🎛️ Optional: Filter Data")
    with st.expander("Filter with dropdowns"):
        col1 = st.selectbox("Select column", df.columns)
        op1 = st.selectbox("Select operator", ["==", "!=", ">", "<", ">=", "<="])
        val1 = st.text_input("Enter value")

        col2, op2, val2, connector = None, None, None, None
        if st.checkbox("Add second condition"):
            col2 = st.selectbox("Select second column", df.columns)
            op2 = st.selectbox("Select second operator", ["==", "!=", ">", "<", ">=", "<="])
            val2 = st.text_input("Enter second value")
            connector = st.selectbox("Connector", ["and", "or"])

        if st.button("Apply Filter"):
            try:
                val1_clean = f"'{val1}'" if not is_numeric(val1) else val1
                query = f"`{col1}` {op1} {val1_clean}"
                if col2 and op2 and val2:
                    val2_clean = f"'{val2}'" if not is_numeric(val2) else val2
                    query += f" {connector} `{col2}` {op2} {val2_clean}"
                st.code(query)
                filtered = df.query(query)
                st.dataframe(filtered if not filtered.empty else "No matching rows.")
            except Exception as e:
                st.error(f"Error: {e}")

    # 📈 Chart Section
    st.markdown("### 📈 Optional: Visualize Chart")
    with st.expander("Create a chart"):
        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Pie"])
        x_col = st.selectbox("X-axis", df.columns)
        if chart_type != "Pie":
            y_cols = st.multiselect("Y-axis (select numeric columns)", df.select_dtypes(include='number').columns)
        else:
            y_cols = []

        if st.button("📊 Generate Chart"):
            try:
                if chart_type == "Bar":
                    st.bar_chart(df.set_index(x_col)[y_cols])
                elif chart_type == "Line":
                    st.line_chart(df.set_index(x_col)[y_cols])
                elif chart_type == "Pie":
                    pie_data = df[x_col].value_counts()
                    st.pyplot(pie_data.plot.pie(autopct="%1.1f%%", figsize=(6,6)).get_figure())
            except Exception as e:
                st.error(f"Error generating chart: {e}")
else:
    st.info("👈 Upload a CSV to get started.")
