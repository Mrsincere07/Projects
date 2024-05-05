import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

st.title("Sentiment Analysis App")

text = st.text_input("Enter text to analyze:")
if st.button("Analyze"):
    encoding = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, attention_mask)
    prediction = int(torch.argmax(output.logits))

    if prediction == 0:
        st.write("Negative")
    elif prediction == 1:
        st.write("Neutral")
    else:
        st.write("Positive")

    values = [output.logits[0][0].item(), output.logits[0][1].item(), output.logits[0][2].item()]
    labels = ["Negative", "Neutral", "Positive"]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)
