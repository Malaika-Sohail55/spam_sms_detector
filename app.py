import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# Load model
MODEL_NAME = "malaika-sohail/spam_sms"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# Streamlit UI
st.title("📩 SMS Spam Classifier")
st.write("Enter a message to check whether it is Spam or Ham.")

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

        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

        confidence = probs[0][prediction].item() * 100

        label = "SPAM 🚨" if prediction == 1 else "HAM ✅"

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}%")
