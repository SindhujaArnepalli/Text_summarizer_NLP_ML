import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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


# -----------------------------
# MODEL LOADING (cached)
# -----------------------------
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # ✅ lightweight version

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()


# -----------------------------
# SUMMARIZATION FUNCTIONS
# -----------------------------
def bart_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=150,
        min_length=30,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def textrank_summary(text, sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])


# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("Customer Review Summarizer 📝")

review = st.text_area("Enter customer review here:")

if st.button("Summarize"):
    if review.strip():
        bart_result = bart_summary(review)
        textrank_result = textrank_summary(review)

        st.subheader("🔹 BART Summary (Abstractive)")
        st.write(bart_result)

        st.subheader("🔹 TextRank Summary (Extractive)")
        st.write(textrank_result)
    else:
        st.warning("Please enter some text first.")
