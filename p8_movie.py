# %%
import pandas as pd
import numpy as np
import re
import nltk

# %%
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# %%
nltk.download('stopwords')


data = pd.read_csv("/Users/siddharthkumar2023/Downloads/NLP_SIDDHARTH_PRACTICALS/IMDB Dataset.csv")

data = data.sample(5000, random_state=42)
print(data.head())

# %%
def preprocess_text(text):

    text = text.lower()


    text = re.sub(r'<.*?>', '', text)


    text = re.sub(r'[^a-zA-Z]', ' ', text) #removing any special


    text = re.sub(r'\s+', ' ', text) #remove extra space

    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)
    # data['review'] = data['review'].apply(preprocess_text)



# %%
#pre-processing
data['review'] = data['review'].apply(preprocess_text)

# %%


# %%

data['sentiment'] = data['sentiment'].map({
    'positive': 1,
    'negative': 0
})

# %%
X = data['review']
y = data['sentiment']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# %%
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression()

model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)

# %%
# from sklearn.naive_bayes import MultinomialNB

# model = MultinomialNB()

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# %%
def predict_sentiment(review):


    review = preprocess_text(review)


    review_vector = tfidf.transform([review])

    prediction = model.predict(review_vector)


    if prediction[0] == 1:
        print("Prediction: Positive")
    else:
        print("Prediction: Negative")

# %%
review1 = "This movie was amazing and full of emotions."
predict_sentiment(review1)

review2 = "Worst movie I have ever watched."
predict_sentiment(review2)

# %%
# import tensorflow as tf

# dataset, info = tf.keras.datasets.imdb.load_data(num_words=10000)

# (X_train, y_train), (X_test, y_test) = dataset

# print("Training samples:", len(X_train))
# print("Label example:", y_train[0])   # 1 = positive, 0 = negative

# Prediction: Positive
# Prediction: Negative



