# %%
# pip install nltk

# %%
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('words')
nltk.download('maxent_ne_chunker_tab')

# %%
import nltk
headlines=[
    " Donuld Trump launches attack on Iran",
    "Kohli scored a century against Pakistan",
    "Apple  launched a new iphone in California",
    "Siddharth is working in TCS in Pune"
]


# %%
for sentences in headlines:
  print(" \nSentences",sentences)
  words=nltk.word_tokenize(sentences)
  pos_tag=nltk.pos_tag(words)
  NER_tree=nltk.ne_chunk(pos_tag)
  print("Named Entities are :")
  for subtree in NER_tree:
    if hasattr(subtree,'label'):
      entity=" ".join([word for word,tag in subtree])
      print(f"{entity}-> {subtree.label()}")

# %%
# Sentences  Donuld Trump launches attack on Iran
# Named Entities are :
# Donuld-> PERSON
# Trump-> PERSON
# Iran-> GPE
 
# Sentences Kohli scored a century against Pakistan
# Named Entities are :
# Kohli-> PERSON
# Pakistan-> GPE
 
# Sentences Apple  launched a new iphone in California
# Named Entities are :
# Apple-> PERSON
# California-> GPE
 
# Sentences Siddharth is working in TCS in Pune
# Named Entities are :
# Siddharth-> GPE
# TCS-> ORGANIZATION
# Pune-> GPE



