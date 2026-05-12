# %%
from collections import Counter
import nltk

# %%
corpus=["NLP is a great subject","ML is a great subject"]

# %%
tokens=" ".join(corpus).split()

# %%
unigram_c=Counter(tokens)

# %%
bigram=list(nltk.bigrams(tokens))
bigram_c=Counter(bigram)

# %%
V=len(unigram_c)
N=len(tokens)

# %%
def unigram_prob(word):
  return (unigram_c[word]+1)/(N +V)

# %%
def bigram_prob(w1,w2):
  return (bigram_c[(w1,w2)]+1)/ (unigram_c[w1]+V)

# %%
sentence= "ML is a great subject".split()

# %%
p_unit=1
for w in sentence:
  p_unit*=unigram_prob(w)

p_bi=unigram_prob(sentence[0])
for i in range (1,len(sentence)):
  p_bi *=bigram_prob(sentence[i-1],sentence[i])
  


# %%
print("Unigram Probability :", p_unit)
print("Bigram Probability  :", p_bi)

# %%
print("Vocabulary Size (V) =", V)

# Total Number of Tokens
print("Total Tokens (N) =", N)


# %%
# Unigram Probability : 0.0001544952392578125
# Bigram Probability  : 0.0018833705357142855

# Vocabulary Size (V) = 6
# Total Tokens (N) = 10


