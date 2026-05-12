# %%
# %%
# Import Libraries


# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# %%
# Load Dataset

data = pd.read_csv("IMDB Dataset.csv")

# %%
# Take Small Sample (Optional)

data = data.sample(5000, random_state=42)

# %%
# Input and Output

X = data['review']

y = data['sentiment']

# %%
# Convert Labels into Numerical Form

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# %%
# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
# Tokenization

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X_train)

# %%
# Convert Text to Sequences

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# %%
# Padding Sequences

X_train_pad = pad_sequences(X_train_seq, maxlen=100)

X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# %%
# Build LSTM Model

model = Sequential()

model.add(
    Embedding(
        input_dim=5000,
        output_dim=64,
        input_length=100
    )
)

model.add(LSTM(64))

model.add(Dense(1, activation='sigmoid'))

# %%
# Compile Model

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# %%
# Train Model

model.fit(
    X_train_pad,
    y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.2
)

# %%
# Evaluate Model

loss, accuracy = model.evaluate(X_test_pad, y_test)

print("Test Accuracy:", accuracy)

# %%
# Prediction Function

def predict_review(review):

    seq = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(seq, maxlen=100)

    pred = model.predict(padded)

    if pred[0][0] > 0.5:
        print("Positive Review")

    else:
        print("Negative Review")

# %%
# Test Predictions

predict_review("This movie was fantastic and emotional")

predict_review("Worst movie ever")


