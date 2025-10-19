import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Загружаем тестовые данные
data = pd.read_csv("data/requests.csv")

# Текст и метки
X = data["text"]
y = data["label"]

# Векторизация текста
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Простая модель классификации заявок
model = MultinomialNB()
model.fit(X_vec, y)

# Проверим на новом запросе
new_text = ["Нужен ремонт квартиры на 50 м2"]
pred = model.predict(vectorizer.transform(new_text))

print("Тип заявки:", pred[0])
