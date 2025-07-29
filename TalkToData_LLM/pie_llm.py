import streamlit as st
import requests

# ---------- App Setup ----------
st.set_page_config(page_title="Premier Data Assistant", layout="centered")
st.title("💼 Premier Data Assistant")
st.markdown("Ask me anything about data governance, quality, or migration at Premier.")

# ---------- Warm-up Model on Startup ----------
@st.cache_resource
def warmup_model():
    try:
        requests.post("http://localhost:11434/api/generate", json={
            "model": "premier-data-llm",
            "prompt": "Warmup",
            "stream": False
        })
    except Exception as e:
        st.warning(f"⚠️ Model warmup failed: {e}")

warmup_model()

# ---------- Main Request Function ----------
def ask_premier(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "premier-data-llm",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "No response received.")
    except Exception as e:
        return f"❌ Error: {e}"

# ---------- UI for Question Input ----------
with st.form("ask_form"):
    user_input = st.text_input(
        "Ask your question here:",
        placeholder="e.g. Who handles migration testing?"
    )
    submitted = st.form_submit_button("Ask Premier")

# ---------- Response Display ----------
if submitted and user_input:
    with st.spinner("Thinking..."):
        answer = ask_premier(user_input)
        st.success(answer)
