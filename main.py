import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib

dataset= pd.read_csv("./data/spam.csv", encoding="latin-1")[['v1', 'v2']]
dataset.columns = ['Label', 'Message']
dataset['Label']=dataset['Label'].map({'ham':0,'spam':1})
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(dataset["Message"])  # Convert text to numerical features
y = dataset["Label"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = SVC(kernel="linear").fit(X_train,y_train)
y_pred=model.predict(X_test)
# print(f"Accuracy:{accuracy_score(y_test,y_pred)*100:.2f}%")
# print(classification_report(y_test, y_pred))
new_messages = ["You have now won free trip to Dubai", "Hey, what's up?"]
new_messages_vectorized = vectorizer.transform(new_messages)

predictions = model.predict(new_messages_vectorized)

for msg, pred in zip(new_messages, predictions):
    print(f"{msg} -> {'Spam' if pred == 1 else 'Ham'}")

joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")