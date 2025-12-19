# train.py
import pandas as pd
import numpy as np
import re, pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# ================= DATA =================
sms = pd.read_csv("data/spam.csv", encoding="latin-1")[['v1','v2']]
sms.columns = ['label','text']
sms['label'] = sms['label'].map({'ham':0,'spam':1})

# OPTIONAL: Enron
try:
    enron = pd.read_csv("data/emails.csv")
    enron = enron[['text','label']]
    data = pd.concat([sms[['text','label']], enron])
except:
    data = sms[['text','label']]

# ================= PREPROCESS =================
stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join([w for w in text.split() if w not in stop_words])

data['clean_text'] = data['text'].apply(clean)

# ================= TOKENIZER =================
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['clean_text'])

X = pad_sequences(tokenizer.texts_to_sequences(data['clean_text']), maxlen=max_len)
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ================= MODEL =================
model = Sequential([
    Embedding(max_words,64,input_length=max_len),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=5,batch_size=32,validation_split=0.1)

# ================= SAVE =================
model.save("model/lstm_model.h5")
pickle.dump(tokenizer, open("model/tokenizer.pkl","wb"))

# ================= EVALUATION =================
y_prob = model.predict(X_test)
y_pred = (y_prob>0.5).astype(int)

print(classification_report(y_test,y_pred))
print("ROC-AUC:", roc_auc_score(y_test,y_prob))

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
