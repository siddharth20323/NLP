# %%
# %%
# Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# %%
# Create Random Dataset

data = pd.DataFrame({

    'review': [

        "This movie was amazing",
        "I loved this film",
        "Fantastic acting and story",
        "Very emotional and beautiful",
        "Best movie ever",

        "Worst movie ever",
        "Very boring film",
        "I hated this movie",
        "Terrible acting",
        "Waste of time"

    ],

    'sentiment': [

        "positive",
        "positive",
        "positive",
        "positive",
        "positive",

        "negative",
        "negative",
        "negative",
        "negative",
        "negative"
    ]
})

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

tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(X_train)

# %%
# Convert Text to Sequences

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# %%
# Padding

X_train_pad = pad_sequences(X_train_seq, maxlen=10)

X_test_pad = pad_sequences(X_test_seq, maxlen=10)

# %%
# Build LSTM Model

model = Sequential()

model.add(
    Embedding(
        input_dim=1000,
        output_dim=32,
        input_length=10
    )
)

model.add(LSTM(32))

model.add(Dense(1, activation='sigmoid'))

# %%
# Compile Model

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# %%
# Train Model

model.fit(
    X_train_pad,
    y_train,
    epochs=10,
    verbose=1
)

# %%
# Evaluate Model

loss, accuracy = model.evaluate(X_test_pad, y_test)

print("Accuracy:", accuracy)

# %%
# Prediction Function

def predict_review(review):

    seq = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(seq, maxlen=10)

    pred = model.predict(padded)

    print("\nReview:", review)

    if pred[0][0] > 0.5:
        print("Positive Review")

    else:
        print("Negative Review")

# %%
# Test Predictions

predict_review("Amazing and emotional movie")

predict_review("Very bad and boring film")

# %%



