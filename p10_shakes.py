# %%
# %%
# Import Libraries

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# %%
# Sample Text Dataset

text = """
Shakespeare was a great writer.
Artificial Intelligence is changing the world.
Machine learning makes systems intelligent.
"""

text = text.lower()

# %%
# Create Character Vocabulary

chars = sorted(list(set(text)))

char_to_int = {c:i for i,c in enumerate(chars)}
int_to_char = {i:c for i,c in enumerate(chars)}

# %%
# Prepare Input and Output Sequences

sequence_length = 10

X = []
y = []

for i in range(len(text) - sequence_length):

    seq_in = text[i:i + sequence_length]

    seq_out = text[i + sequence_length]

    X.append([char_to_int[char] for char in seq_in])

    y.append(char_to_int[seq_out])

# %%
# Reshape and Normalize Input

X = np.reshape(X, (len(X), sequence_length, 1))

X = X / float(len(chars))

# %%
# One-Hot Encode Output

y = to_categorical(y, num_classes=len(chars))

# %%
# Build Character-Based LSTM Model

model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(Dense(len(chars), activation='softmax'))

# %%
# Compile Model

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# %%
# Train Model

model.fit(
    X,
    y,
    epochs=20,
    batch_size=32
)

# %%
# Function to Predict Next Character

def predict_next_char(seed_text):

    pattern = [char_to_int[char] for char in seed_text.lower()]

    x = np.reshape(pattern, (1, len(pattern), 1))

    x = x / float(len(chars))

    prediction = model.predict(x, verbose=0)

    index = np.argmax(prediction)

    result = int_to_char[index]

    print("Input Sequence :", seed_text)
    print("Predicted Character :", result)

# %%
# Test Prediction

predict_next_char("shakespear")

# %%



