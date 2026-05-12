# %%


# %%
import nltk
import re
import pandas as pd
import numpy as np

# %%
nltk.download("punkt_tab")
nltk.download("stopwords")

# %%
pip install spacy

# %%
import spacy

# %%
news_corpus="""Artificial Intelligence is changing the world. 
AI helps in healthcare, education, and business. 
At the same time, issues like privacy and bias are important."""

# %%


# %%
original_text=print(news_corpus)

# %%
from nltk.tokenize import word_tokenize

# %%
tokens=word_tokenize(news_corpus)
print("Tokens :")
print(tokens)

# %%
token_lower=[word.lower() for word in tokens if word.isalpha()]
print("after lc and punc removal")
print(token_lower)

# %%
from nltk.corpus import stopwords
stop_words = set (stopwords. words ('english'))
filtered_tokens = [word for word in token_lower if word not in stop_words]

# %%
print("after stopword:")
print(filtered_tokens)

# %%
from nltk. stem import PorterStemmer
stemmer = PorterStemmer ()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("After Stemming:")
print (stemmed_tokens)

# %%
!python -m spacy download en_core_web_sm

# %%
nlp = spacy.load("en_core_web_sm")
doc = nlp(" ". join(filtered_tokens))
lemmatized_tokens = [token. lemma_ for token in doc]
print ( "After Lemmatization:")
print (lemmatized_tokens)

# %%
print(len(lemmatized_tokens))

# %%
print(len(stemmed_tokens))

# %%



# Tokens :
# ['Artificial', 'Intelligence', 'is', 'changing', 'the', 'world', '.', 'AI', 'helps', 'in', 'healthcare', ',', 'education', ',', 'and', 'business', '.', 'At', 'the', 'same', 'time', ',', 'issues', 'like', 'privacy', 'and', 'bias', 'are', 'important', '.']

# after lc and punc removal
# ['artificial', 'intelligence', 'is', 'changing', 'the', 'world', 'ai', 'helps', 'in', 'healthcare', 'education', 'and', 'business', 'at', 'the', 'same', 'time', 'issues', 'like', 'privacy', 'and', 'bias', 'are', 'important']

# after stopword:
# ['artificial', 'intelligence', 'changing', 'world', 'ai', 'helps', 'healthcare', 'education', 'business', 'time', 'issues', 'like', 'privacy', 'bias', 'important']

# After Stemming:
# ['artifici', 'intellig', 'chang', 'world', 'ai', 'help', 'healthcar', 'educ', 'busi', 'time', 'issu', 'like', 'privaci', 'bia', 'import']

# 15
# 15