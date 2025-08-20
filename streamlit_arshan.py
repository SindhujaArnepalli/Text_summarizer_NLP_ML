import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch 
import nltk

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")  # for latest versions

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Load model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # ensure real tensors
    device_map=None              # prevent meta device
)

device = torch.device("cpu")
model.to(device)

st.title("Customer Review Summarizer")

# User input
review = st.text_area("Enter customer review here:")

def bart_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=4,
        max_length=150,
        min_length=40,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def textrank_summary(text, sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])

if st.button("Summarize"):
    if review.strip():
        bart_result = bart_summary(review)
        textrank_result = textrank_summary(review)

        st.subheader("🔹 BART Summary (Abstractive)")
        st.write(bart_result)

        st.subheader("🔹 TextRank Summary (Extractive)")
        st.write(textrank_result)
