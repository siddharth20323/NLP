from textblob.classifiers import NaiveBayesClassifier

# %%
# Training Data

train = [
    ('I love this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
]

# %%
# Train Classifier

classifier = NaiveBayesClassifier(train)

# %%
# Test Sentence

text = "The food was wonderful."

result = classifier.classify(text)

# %%
print("Sentence:", text)
print("Prediction:", result)