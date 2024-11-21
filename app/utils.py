# app/utils.py

import tensorflow as tf
import pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import current_app
import pandas as pd

# Inisialisasi variabel global
model_intent = None
model_ner = None
tokenizer = None
label_encoder = None
ner_label_encoder = None
vectorizer = None
tfidf_matrix = None
df_utterances = None
ner_label_decoder = None

def load_resources():
    global model_intent, model_ner, tokenizer, label_encoder, ner_label_encoder, vectorizer, tfidf_matrix, df_utterances, ner_label_decoder

    # Load model menggunakan tf.keras
    model_intent = tf.keras.models.load_model('app/models/model_intent.keras')
    model_ner = tf.keras.models.load_model('app/models/model_ner.keras')

    # Load encoders dan tokenizer
    with open('app/encoders/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('app/encoders/label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)

    with open('app/encoders/ner_label_encoder.pickle', 'rb') as handle:
        ner_label_encoder = pickle.load(handle)
        ner_label_decoder = {v: k for k, v in ner_label_encoder.items()}

    # Load vectorizer dan TF-IDF matrix
    with open('app/data/vectorizer.pickle', 'rb') as handle:
        vectorizer = pickle.load(handle)

    with open('app/data/tfidf_matrix.pickle', 'rb') as handle:
        tfidf_matrix = pickle.load(handle)

    # Load DataFrame respons
    df_utterances = pd.read_pickle('app/data/df_utterances.pkl')

def clean_text(text):
    """Membersihkan teks input."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_intent(text):
    """Memprediksi intent dari teks input."""
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    max_seq_length = 28  # Sesuaikan dengan maxlen saat pelatihan
    padded_seq = pad_sequences(seq, maxlen=max_seq_length, padding='post')
    pred = model_intent.predict(padded_seq)
    predicted_label = np.argmax(pred, axis=1)[0]
    intent = label_encoder.inverse_transform([predicted_label])[0]
    return intent

def predict_entities(text):
    """Memprediksi entitas dalam teks input."""
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    max_seq_length = 28  # Sesuaikan dengan maxlen saat pelatihan
    padded_seq = pad_sequences(seq, maxlen=max_seq_length, padding='post')
    pred = model_ner.predict(padded_seq)
    pred_labels = np.argmax(pred, axis=-1)[0]
    tokens = text_clean.split()
    entities = []
    for idx, label_id in enumerate(pred_labels[:len(tokens)]):
        label = ner_label_decoder.get(label_id, 'O')
        if label != 'O':
            entity_type = label.split('-')[1]
            entities.append({'entity': entity_type, 'value': tokens[idx]})
    return entities

def get_response(user_input):
    """Mengambil respons berdasarkan kesamaan cosine."""
    user_input_clean = clean_text(user_input)
    user_tfidf = vectorizer.transform([user_input_clean])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    most_similar_idx = np.argmax(similarities[0])
    if similarities[0][most_similar_idx] < 0.35:
        return "Maaf, saya tidak memahami pertanyaan Anda."
    response = df_utterances.iloc[most_similar_idx]['responses']
    return response

def chatbot_response(user_input):
    """Menggabungkan prediksi intent, entitas, dan mendapatkan respons."""
    intent = predict_intent(user_input)
    entities = predict_entities(user_input)
    response = get_response(user_input)
    return response

# Load resources saat modul diimpor
load_resources()
