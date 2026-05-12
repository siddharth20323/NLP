# %%
import nltk
import re
import pandas as pd
import numpy as np

# %%
from nltk.corpus import gutenberg,stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer



# %%


# %%
nltk.download("gutenberg")
nltk.download("punkt_tab")
nltk.download("stopwords")

# %%


# %%
emma_text=gutenberg.raw('austen-emma.txt')

# %%
sentences=sent_tokenize(emma_text)

# %%
corpus=sentences[:10]

# %%
corpus

# %%
stop_words=set(stopwords.words("english"))

# %%

def normalize_doc(doc):

    # Remove punctuation and numbers
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)

    # Convert to lowercase
    doc = doc.lower()

    # Tokenization
    tokens = doc.split()

    # Remove stopwords
    filtered_tokens = [
        token for token in tokens
        if token not in stop_words
    ]

    # Join tokens again
    return " ".join(filtered_tokens)



# %%


# %%
normalized_corpus = [
    normalize_doc(doc)
    for doc in corpus
]

# %%


# %%


# %%
from sklearn.feature_extraction.text import CountVectorizer


# %%
cv=CountVectorizer(min_df=0.,max_df=1.)

# %%
cv_matrix=cv.fit_transform(normalized_corpus)

# %%
cv_matrix

# %%
print(cv_matrix)

# %%
cv.get_feature_names_out()

# %%
cv_matrix=cv_matrix.toarray()

# %%
cv_matrix

# %%
vocab=cv.get_feature_names_out()

# %% [markdown]
# 

# %%
df=pd.DataFrame(cv_matrix,columns=vocab)

# %%
df

# %%
word_t=df.sum(axis=0)
word_t

# %%
most_com=word_t.sort_values(ascending=False)
most_com

# %%
print("Most Common Words:\n")
print(most_com.head(10))

# %%
most_com_df=most_com.reset_index()
most_com_df.columns=["Word","Frequency"]
most_com_df.head(10)

# %%

# Word	Frequency
# 0	miss	4
# 1	emma	4
# 2	taylor	3
# 3	little	3
# 4	governess	3
# 5	friend	3
# 6	taylors	2
# 7	daughters	2
# 8	mother	2
# 9	disposition	2




