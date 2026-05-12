import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
# Sample Data

sentences = [
    "I love it",
    "I hate it",
    "Fantastic",
    "Awful"
]

labels = np.array([1, 0, 1, 0])

# %%
# Tokenization

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding='post')

# %%
# Build Model

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(
        input_dim=100,
        output_dim=16,
        input_length=padded.shape[1]
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(24, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

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
    padded,
    labels,
    epochs=10,
    verbose=0
)

# %%
# Prediction

test_sentence = ["I love this!"]

test_seq = tokenizer.texts_to_sequences(test_sentence)

test_padded = pad_sequences(
    test_seq,
    maxlen=padded.shape[1],
    padding='post'
)

prediction = model.predict(test_padded)

print("Prediction Score:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")