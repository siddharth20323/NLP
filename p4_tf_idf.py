# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# %%
documents = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love Python programming"
]

# %%
vectorizer = TfidfVectorizer()

# %%
tfidf_matrix = vectorizer.fit_transform(documents)

# %%
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=["Doc1", "Doc2", "Doc3"]
)

# %%
print("TF-IDF Matrix:\n")
print(df)



idf_values = vectorizer.idf_

idf_df = pd.DataFrame({
    "Term": terms,
    "IDF": idf_values
})

print("\nIDF Values:\n")
print(idf_df)

# %%
# TF Calculation

print("\nTF Values:\n")

for i, doc in enumerate(documents):

    words = doc.lower().split()
    total_words = len(words)

    print(f"\nDocument {i+1}")

    for term in terms:

        tf = words.count(term) / total_words

        if tf > 0:
            print(f"{term} : {tf:.3f}")
# %%
# TF IDF-MATRIX
#        amazing        is  learning     love   machine  programming    python
# DOC1  0.000000  0.000000  0.577350  0.57735  0.577350     0.000000  0.000000
# DOC2  0.562829  0.562829  0.428046  0.00000  0.428046     0.000000  0.000000
# Doc3  0.000000  0.000000  0.000000  0.47363  0.000000     0.622766  0.622766

# IDF Values:

#           Term       IDF
# 0      amazing  1.693147
# 1           is  1.693147
# 2     learning  1.287682
# 3         love  1.287682
# 4      machine  1.287682
# 5  programming  1.693147
# 6       python  1.693147

# TF Values:


# Document 1
# learning : 0.250
# love : 0.250
# machine : 0.250

# Document 2
# amazing : 0.250
# is : 0.250
# learning : 0.250
# machine : 0.250

# Document 3
# love : 0.250
# programming : 0.250
# python : 0.250






