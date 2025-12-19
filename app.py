# ===== Streamlit UI: Fraud Detection (Inference Only) =====
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ===== Load Model & Tokenizer =====
@st.cache_resource
def load_artifacts():
    model = load_model("model/lstm_model.h5")
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# ===== Text Preprocessing =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ===== UI =====
st.title("SMS Fraud Detection (NLP + LSTM)")
st.write("Deteksi apakah pesan termasuk **Penipuan (Fraud)** atau **Normal (Legitimate)**")

user_input = st.text_area("Masukkan teks SMS / pesan:")

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong")
    else:
        clean = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=100)
        pred = model.predict(padded)[0][0]

        if pred > 0.5:
            st.error(f"FRAUD / SPAM (Probabilitas: {pred:.2f})")
        else:
            st.success(f"LEGITIMATE (Probabilitas: {1-pred:.2f})")
