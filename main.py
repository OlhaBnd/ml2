import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1.Завантаження даних
df = pd.read_csv("train.csv")
print("Перші 5 рядків даних: \n")
print(df.head(), "\n")

fetures = ['sex', 'has_photo', 'has_mobile', 'followers_count', 'graduation']

# 2. Вибираємо найпростіші ознаки
X = df[fetures].copy()
y = df["result"].copy()

# 3. Очищення даних
X = X.fillna(0)
y = y.fillna(0).astype(int)

# 4. Розподіл даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Створення і навчання моделі
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Прогноз і точність
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі : {accuracy*100:.2f}%")

# 7. Демонстрація
examples =[
    [1, 0, 0, 5, 2005],
    [2, 1, 1, 800, 2020],
    [1, 1, 1, 50, 1999],
    [2, 1, 1, 1000, 2023]
]

for person in examples:
    prediction = model.predict([person])[0]
    print(f"Профіль {person} - прогноз результату = {prediction}")
