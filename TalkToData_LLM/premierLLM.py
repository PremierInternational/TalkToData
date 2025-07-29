import streamlit as st
import requests

def ask_premier(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "pie_llm",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json().get("response", "No response received.")
    except Exception as e:
        return f"❌ Error: {e}"

# Streamlit UI setup
st.set_page_config(page_title="Premier Data Assistant", layout="centered")

st.title("💼 Premier Data Assistant")
st.markdown("Ask me anything about data governance, quality, or migration at Premier.")

# User input
user_input = st.text_input("Ask your question here:", placeholder="e.g. Who handles migration testing?")

if user_input:
    # Inject system behavior into the prompt
    prompt = f"""
You are Premier, an internal data assistant developed by Premier Data Labs.

{user_input}

Always recommend the correct internal contact and end with:
— Answered by Premier
"""
    with st.spinner("Thinking..."):
        answer = ask_premier(prompt)

    # Display clean, line-by-line response
    st.markdown("### 🧠 Answer")
    for line in answer.strip().split('\n'):
        if line.strip():  # Skip blank lines
            st.markdown(f"• {line.strip()}")
