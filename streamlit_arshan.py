import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import nltk

# Ensure punkt is available (sumy needs only 'punkt')
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

st.title("Customer Review Summarizer")

# ---- Load model/tokenizer on CPU, avoiding meta tensors ----
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

# Key fixes:
# 1) low_cpu_mem_usage=False -> prevents meta initialization
# 2) map_location="cpu"       -> ensure weights land on CPU
# 3) do NOT pass device_map="auto" (can trigger meta in some combos)
model = BartForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=False,
    torch_dtype=torch.float32
)
device = torch.device("cpu")
model.to(device)
model.eval()

# ---- UI ----
review = st.text_area("Enter customer review here:")

def bart_summary(text: str) -> str:
    inputs = tokenizer(
        [text],
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    # move Tensors to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=150,
            min_length=40,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def textrank_summary(text: str, sentences_count: int = 2) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(s) for s in summary)

if st.button("Summarize"):
    text = (review or "").strip()
    if not text:
        st.warning("Please paste a review first.")
    else:
        # Run BART safely; fall back to TextRank if anything fails
        bart_result = None
        try:
            bart_result = bart_summary(text)
        except Exception as e:
            st.error(f"BART failed: {e}")

        textrank_result = textrank_summary(text)

        if bart_result:
            st.subheader("🔹 BART Summary (Abstractive)")
            st.write(bart_result)

        st.subheader("🔹 TextRank Summary (Extractive)")
        st.write(textrank_result)
