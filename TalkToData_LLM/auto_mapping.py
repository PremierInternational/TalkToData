#author : Sanjay
#version : v1.0
#date : June-2025


import streamlit as st
import pandas as pd
import json
import subprocess
import re

st.set_page_config(page_title="Column Profiling and Mapping", layout="wide")
st.title("🧠 Auto Target Column Suggestion (Local Ollama + Streamlit)")
src_file = st.file_uploader("Upload Source Dataset", type="csv", key="src")

if src_file:
    src_df = pd.read_csv(src_file)

    st.subheader("📊 Source Dataset Preview")
    st.dataframe(src_df.head(5))

    if st.button("🔍 Suggest Target Columns and Business Metadata"):
        
        prompt = f"""
        You are a data modeling assistant.

        Given the following source dataset columns and sample data:

        Columns: {list(src_df.columns)}
        Sample data:
        {src_df.head(3).to_dict()}

        For each source column, suggest:
        - Suggested target column name (more meaningful)
        - Data type
        - Business term (description of what the column likely represents)

        Return your response in this format:
        [
          {{"source": "source_column_name", "target": "target_column_name", "type": "DataType", "description": "Business meaning"}},
          ...
        ]
        """

        response = subprocess.run(
            ["ollama", "run", "mistral"], input=prompt.encode(), stdout=subprocess.PIPE
        )
        raw_output = response.stdout.decode()

        json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)

        if json_match:
            try:
                suggestions = json.loads(json_match.group())
                st.subheader("📋 Suggested Target Mappings and Metadata")
                st.dataframe(pd.DataFrame(suggestions))

                if st.button("💾 Download Mapping Suggestions as JSON"):
                    json_str = json.dumps(suggestions, indent=2)
                    st.download_button("Download Suggestions", json_str, file_name="target_column_suggestions.json", mime="application/json")

            except Exception as e:
                st.error("❌ Failed to parse JSON. Full model output:")
                st.text(raw_output)
        else:
            st.error("Model output did not include a valid JSON array. Full response:")
            st.text(raw_output)
else:
    st.info("Please upload a source dataset to begin.")
