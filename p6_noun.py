# %%
pip install nltk

# %%
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

# %%
import nltk
from collections import Counter
with open("input1.txt","r") as file:
  text=file.read()

# %%
words=nltk.word_tokenize(text)

# %%
pos_tag=nltk.pos_tag(words)

# %%
print("POS tag words are: ")
print(pos_tag)

# %%
nouns =[]
for word, tag in pos_tag:
  if tag in ['NN',"NNS","NNP","NNPS"]:
    nouns.append(word)

# %%
print("All NOUNS in the sentences are :")
print(nouns)

# %%
# for counting frequency of POS tags
pos_counts=Counter(tag for word, tag in pos_tag)

# %%
print(" POS tag frequecy are :")
print(dict(pos_counts))

# %%


# POS tag words are: 
# [('Siddharth', 'NNP'), ('is', 'VBZ'), ('working', 'VBG'), ('in', 'IN'), ('tcs', 'NN'), ('at', 'IN'), ('noida', 'NN')]
# All NOUNS in the sentences are :
# ['Siddharth', 'tcs', 'noida']
# POS tag frequecy are :
# {'NNP': 1, 'VBZ': 1, 'VBG': 1, 'IN': 2, 'NN': 2}

