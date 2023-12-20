import streamlit as st
from transformers import pipeline
from bert import run_bert

st.title("CNN-QnA")
sentence = st.text_area('Please paste your article :', height=30)
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")

with st.spinner("Discovering Answers.."):
    if button and sentence:
        if question == "":
            st.warning("Question is empty, please put question before submit.")
        else:
            answers = run_bert(question, sentence)
            st.write(answers)

