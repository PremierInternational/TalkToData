import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import streamlit as st
import pandas as pd
import re
import requests
import time
from difflib import get_close_matches
import validators
from typing import Tuple, Optional, Union
import logging
import io
from pathlib import Path
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logo_base64 = get_base64_image("definian.png")
# ──────────────────────────────
# Streamlit config
# ──────────────────────────────
st.set_page_config(page_title="NLP Data Analysis with Smart Filters", layout="wide")

st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_base64}" alt="Company Logo" style="height: 60px; margin-right: 15px;">
        <h1 style="color: #006400;"></h1>
    </div>
""", unsafe_allow_html=True)

# Custom CSS for enhanced UI with blue background and white text
st.markdown("""
    <style>
    /* General */
    body {
        font-family: 'Segoe UI', sans-serif;
        color: #003366;  /* Dark blue for text */
    }

    .stApp {
        background-color: #ffffff; /* Alice blue */
    }

    .stTitle {
        color: #006400;  /* Dark green title */
        font-weight: bold;
        margin-bottom: 10px;
    }

    .stSubheader {
        color: #2e8b57;  /* Medium sea green */
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Text input */
    .stTextInput > div > div > input {
        border: 2px solid #1e90ff; /* Dodger blue border */
        border-radius: 6px;
        padding: 10px;
        background-color: #ffffff;
        color: #003366;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2e8b57; /* Medium sea green */
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #006400; /* Dark green */
    }

    /* Metrics */
    .stMetric {
        background-color: #e6f2ff; /* Very light blue */
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: 1px solid #1e90ff; /* Dodger blue */
        color: #003366;
    }

    /* DataFrame */
    .stDataFrame {
        border: 2px solid #2e8b57;
        border-radius: 6px;
        background-color: #f5fff9; /* Soft green tint */
        padding: 10px;
        margin-top: 10px;
        color: #003366;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #dbefff; /* Light sky blue */
        color: #003366;
    }

    .sidebar .stSelectbox, .sidebar .stCheckbox {
        background-color: #e6ffe6; /* Pale green */
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 15px;
        border: 1px solid #2e8b57;
        color: #003366;
    }

    /* Spinner */
    .stSpinner > div > div {
        color: #2e8b57;
    }

    /* Markdown headers */
    .stMarkdown h3, .stMarkdown h4 {
        color: #006400;
    }

    /* Checkbox label */
    .stCheckbox > label {
        color: #003366;
    }
    </style>
""", unsafe_allow_html=True)


st.title("   Data Analysis with NLP and Smart Filters")

# Cache the DataFrame to avoid reloading
@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

# File uploader
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"], key="file_uploader")

# ──────────────────────────────
# Ollama API call
# ──────────────────────────────
@st.cache_data(ttl=3600)
def query_ollama(prompt: str, model: str = "tinyllama") -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return f"Error: {e}"

# ──────────────────────────────
# Smart Query Helpers
# ──────────────────────────────
def compact(s: str) -> str:
    return re.sub(r'\W+', '', str(s).lower())

def find_best_column(question: str, columns: list) -> Optional[str]:
    q_compact = compact(question)
    q_lower = question.lower()
    
    # Exact match
    for col in columns:
        if compact(col) and compact(col) in q_compact:
            return col
    
    # Token-based match
    for col in columns:
        tokens = re.findall(r'\w+', str(col).lower())
        if not tokens:
            continue
        pattern = r'\W*'.join(re.escape(t) for t in tokens)
        if re.search(pattern, q_lower):
            return col
    
    # Fuzzy match
    col_compacts = [compact(c) for c in columns]
    words = re.findall(r'\w+', q_lower)
    for w in words:
        matches = get_close_matches(w, col_compacts, n=1, cutoff=0.85)
        if matches:
            idx = col_compacts.index(matches[0])
            return columns[idx]
    
    return None

def extract_condition(question: str, found_col: str) -> Tuple[Optional[str], Optional[Union[str, float, int]]]:
    q = question.lower()

    if re.search(r'\bduplicate(s)?\b', q):
        return 'duplicate', None

    if ('email' in q) and any(word in q for word in ["invalid", "incorrect", "wrong", "bad", "malformed"]):
        return 'invalid_email', None

    # Domain extraction with validation
    m = re.search(r'@([a-z0-9\.-]+\.[a-z]{2,})', question, flags=re.I)
    if m and validators.domain(m.group(1)):
        return 'domain', m.group(1)
    m = re.search(r'domain\s+["\']?([a-z0-9\.-]+\.[a-z]{2,})["\']?', q)
    if m and validators.domain(m.group(1)):
        return 'domain', m.group(1)

    if re.search(r'\b(missing|null|empty|blank)\b', q):
        return 'missing', None

    if re.search(r'\b(digits|characters)\b', q):
        num = re.search(r'(\d+)', q)
        if num:
            n = int(num.group(1))
            if re.search(r'(more than|greater than|>)', q):
                return 'digits_more', n
            if re.search(r'(less than|below|<)', q):
                return 'digits_less', n

    m = re.search(r'(starts with|start with)\s+["\']?(?P<val>.+)', q)
    if m:
        return 'startswith', m.group('val').strip().strip('"\'')
    
    m = re.search(r'(ends with|ends|last is)\s+["\']?(?P<val>.+)', q)
    if m:
        return 'endswith', m.group('val').strip().strip('"\'')
    
    m = re.search(r'(?:contains|has|include|includes|have)\s+["\']?(?P<val>.+)', q)
    if m:
        return 'contains', m.group('val').strip().strip('"\'')
    
    m = re.search(r'(?:equal to|equals|=)\s+["\']?(?P<val>.+)', q)
    if m:
        return 'equal', m.group('val').strip().strip('"\'')

    m = re.search(r'(?:not equal to|!=|is not|<>|not equals?)\s+["\']?(?P<val>.+)', q)
    if m:
        return 'not_equal', m.group('val').strip().strip('"\'')



    if re.search(r'\bis\b', q):
        m2 = re.search(r'is\s+["\']?(?P<val>.+)', q)
        if m2:
            return 'equal', m2.group('val').strip().strip('"\'')
    
    m = re.search(r'(?:more than|greater than|above|is more than|is greater than|>)\s*(?P<num>[-+]?\d+(\.\d+)?)', q)
    if m:
        return 'gt', float(m.group('num'))
    m = re.search(r'(?:less than|below|<)\s*(?P<num>[-+]?\d+(\.\d+)?)', q)
    if m:
        return 'lt', float(m.group('num'))
    m = re.search(r'(?:at least|>=|greater than or equal to)\s*(?P<num>[-+]?\d+(\.\d+)?)', q)
    if m:
        return 'ge', float(m.group('num'))
    m = re.search(r'(?:at most|<=|less than or equal to)\s*(?P<num>[-+]?\d+(\.\d+)?)', q)
    if m:
        return 'le', float(m.group('num'))

    quoted = re.search(r'["\'](?P<v>[^"\']+)["\']', question)
    if quoted:
        return 'equal', quoted.group('v').strip()

    return None, None

def apply_filter(df: pd.DataFrame, col: str, op: str, val: Optional[Union[str, float, int]]) -> pd.DataFrame:
    try:
        cs = df[col]
        if op == 'invalid_email':
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            cs_clean = cs.fillna('').astype(str).str.strip()
            non_null_mask = ~cs.isna() & (cs_clean != '')
            valid_mask = cs_clean.str.match(pattern, na=False)
            invalid_values = cs_clean[non_null_mask & ~valid_mask].tolist()
            logger.info(f"Invalid email values detected: {invalid_values}")
            return df[non_null_mask & ~valid_mask]
        elif op == 'duplicate':
            return df[df.duplicated(subset=[col], keep=False)]
        elif op == 'domain':
            domain = val.lower().lstrip('@')
            cs_str = cs.astype(str).str.strip()
            return df[cs_str.str.lower().str.endswith('@' + domain, na=False)]
        elif op == 'missing':
            cs_str = cs.astype(str).str.strip()
            return df[cs.isna() | (cs_str == '') | (cs_str == 'nan')]
        elif op == 'digits_more':
            cs_str = cs.astype(str).str.strip()
            digits = cs_str.str.replace(r'\D', '', regex=True)
            return df[digits.str.len() > int(val)]
        elif op == 'digits_less':
            cs_str = cs.astype(str).str.strip()
            digits = cs_str.str.replace(r'\D', '', regex=True)
            return df[digits.str.len() < int(val)]
        elif op == 'startswith':
            cs_str = cs.astype(str).str.strip()
            return df[cs_str.str.lower().str.startswith(str(val).lower(), na=False)]
        elif op == 'endswith':
            cs_str = cs.astype(str).str.strip()
            return df[cs_str.str.lower().str.endswith(str(val).lower(), na=False)]
        elif op == 'contains':
            cs_str = cs.astype(str).str.strip()
            return df[cs_str.str.lower().str.contains(str(val).lower(), na=False, regex=False)]
        elif op == 'equal':
            try:
                num_val = float(val)
                col_num = pd.to_numeric(cs, errors='coerce')
                return df[col_num == num_val]
            except Exception:
                cs_str = cs.astype(str).str.strip()
                return df[cs_str.str.lower() == str(val).lower()]
        elif op == 'not_equal':
            try:
                num_val = float(val)
                col_num = pd.to_numeric(cs, errors='coerce')
                return df[col_num != num_val]
            except Exception:
                 cs_str = cs.astype(str).str.strip()
                 return df[cs_str.str.lower() != str(val).lower()]

        elif op in ('gt', 'lt', 'ge', 'le'):
            col_num = pd.to_numeric(cs, errors='coerce')
            if op == 'gt': return df[col_num > float(val)]
            if op == 'lt': return df[col_num < float(val)]
            if op == 'ge': return df[col_num >= float(val)]
            if op == 'le': return df[col_num <= float(val)]
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Filter application error: {e}")
        return pd.DataFrame()

def process_question(question: str, df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    if not question.strip():
        return "❌ Provide a question containing a column name and a condition."
    found_col = find_best_column(question, df.columns)
    if not found_col:
        return f"❌ Could not find any column mentioned. Available: {list(df.columns)}"
    op, val = extract_condition(question, found_col)
    if not op:
        return "❌ Could not interpret the condition."
    result = apply_filter(df, found_col, op, val)
    if result.empty:
        return "No matching rows found."
    return result

# ──────────────────────────────
# Main App
# ──────────────────────────────
if uploaded_file:
    df = load_data(uploaded_file)
    if df.empty:
        st.error("No data loaded. Please check your CSV file.")
    else:
        # Sidebar for settings
        with st.sidebar:
            st.subheader("⚙️ Settings")
            model_choice = st.selectbox("Select Ollama Model", ["tinyllama", "llama3", "mistral"], key="model_choice")
            max_rows = st.slider("Max Rows for Query Results", 5, 100, 10, key="max_rows")
            export_format = st.selectbox("Export Format", ["CSV", "Excel"], key="export_format")

            # Suggested questions for Ollama AI
            #st.markdown("### 📝 Suggested Questions")
            #st.markdown("""
             #   - "What are the most common email domains?"
              #  - "Can you summarize the data for rows where age > 30?"
               # - "Identify duplicate entries in the name column."
                #- "What is the average value of the salary column?"
                #- "Find rows with missing values in the email column."
            #""")

        # Show metrics
        st.subheader("📊 Data Overview")
        duplicate_option = st.selectbox("Check duplicates by:", ["All Columns"] + list(df.columns), key="dup_option")
        if duplicate_option == "All Columns":
            dup_count = df.duplicated(keep=False).sum()
        else:
            dup_count = df.duplicated(subset=[duplicate_option], keep=False).sum()
        
        # Calculate duplicate percentage
        total_rows = len(df)
        dup_percentage = (dup_count / total_rows * 100) if total_rows > 0 else 0
        dup_percentage = f"{dup_percentage:.2f}%"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", total_rows)
        col2.metric("Duplicate Count", dup_count)
        col3.metric("Total Columns", len(df.columns))
        col4.metric("Duplicate Percentage", dup_percentage)

        # Data preview (fixed to 3 rows)
        df_preview = df.copy()
        df_preview.index = range(1, len(df_preview) + 1)
        df_preview.index.name = "S.No"
        st.subheader("📋 Data Preview (First 3 Rows)")
        #st.dataframe(df_preview.head(3), use_container_width=True)
        st.dataframe(df_preview.head(3), width="stretch")


        # Question input
        st.markdown("### 💬 Ask Your Question")
        user_question = st.text_input("Type your question (e.g., 'Show invalid emails in email column')", key="q")
        use_ollama = st.checkbox("🤖 Use Ollama AI for Advanced Queries", key="use_ollama")

        if user_question:
            with st.spinner("Processing your request…"):
                if use_ollama:
                    # Handle Ollama query with a single spinner
                    prompt = (
                        "You are a data analysis expert.\n"
                        f"Columns in the DataFrame: {list(df.columns)}.\n"
                        f"Question: \"{user_question}\"\n"
                        "Provide a concise and accurate answer based on the data."
                    )
                    t0 = time.time()
                    ollama_text = query_ollama(prompt, model=model_choice)
                    t1 = time.time()
                    st.metric("🕒 Ollama Response Time (sec)", round(t1 - t0, 2))
                    st.markdown("### 🤖 Ollama Response")
                    st.write(ollama_text)
                else:
                    # Handle standard query processing
                    result = process_question(user_question, df)
                    st.markdown("### ✅ Answer")
                    if isinstance(result, pd.DataFrame):
                        result_display = result.copy()
                        result_display.index = range(1, len(result_display) + 1)
                        result_display.index.name = "S.No"
                        st.write(f"**{len(result)} matching rows out of {len(df)} total rows**")
                        #st.dataframe(result_display.head(max_rows), use_container_width=True)
                        st.dataframe(result_display.head(max_rows), width="stretch")

                        
                        # Export option
                        if not result.empty:
                            st.markdown("#### 📥 Export Results")
                            if export_format == "CSV":
                                csv = result.to_csv(index=False)
                                st.download_button(
                                    label="Download as CSV",
                                    data=csv,
                                    file_name="filtered_data.csv",
                                    mime="text/csv",
                                    key="download_csv"
                                )
                            else:
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    result.to_excel(writer, index=False)
                                excel_data = output.getvalue()
                                st.download_button(
                                    label="Download as Excel",
                                    data=excel_data,
                                    file_name="filtered_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_excel"
                                )
                    else:
                        st.write(result)
else:
    st.info("👈 Upload a CSV to get started.")