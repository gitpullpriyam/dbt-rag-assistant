import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
from src.generation.generator import answer

st.set_page_config(page_title="dbt RAG Assistant", layout="centered")

st.title(" dbt RAG Assistant")
st.caption("Ask questions about the Jaffle Shop dbt project and get responses with cited sources.")

question = st.text_input("Your question", placeholder="e.g. How is customer_lifetime_value calculated?")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating... Give it a moment!"):
        result = answer(question)

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("Sources"):
        for source in result["sources"]:
            st.code(source)
