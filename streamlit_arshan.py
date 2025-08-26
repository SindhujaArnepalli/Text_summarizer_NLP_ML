import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ------------------------------
# NLTK setup
# ------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ------------------------------
# Model Setup
# ------------------------------
@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"  # Lighter model for Streamlit Cloud
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # ✅ Correct dtype
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ------------------------------
# Streamlit App
# ------------------------------
st.title("📜 Customer Review Summarizer")

review = st.text_area("Enter customer review here:")

def bart_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=150,
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
        with st.spinner("Generating summaries..."):
            bart_result = bart_summary(review)
            textrank_result = textrank_summary(review)

        st.subheader("🔹 BART Summary (Abstractive)")
        st.write(bart_result)

        st.subheader("🔹 TextRank Summary (Extractive)")
        st.write(textrank_result)
    else:
        st.warning("⚠️ Please enter some text to summarize.")
