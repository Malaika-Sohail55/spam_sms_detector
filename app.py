import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from huggingface_hub import hf_hub_download

REPO_ID  = "malaika-sohail/spam_sms"   # fixed: removed trailing slash
BASE_BERT = "bert-base-uncased"

@st.cache_resource                       # loads only once, not on every interaction
def load_model():
    # Load tokenizer from your HF repo (your tokenizer files are fine)
    tokenizer = BertTokenizerFast.from_pretrained(REPO_ID)

    # Load BERT architecture, then inject your saved weights
    model = BertForSequenceClassification.from_pretrained(BASE_BERT, num_labels=2)
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="best_bert_sms.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("📩 SMS Spam Classifier")
st.write("Enter a message to check whether it is **Spam** or **Ham**.")

message = st.text_area("Enter SMS Message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        inputs = tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs      = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item() * 100

        label = "SPAM 🚨" if prediction == 1 else "HAM ✅"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}%")
