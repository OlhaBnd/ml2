import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ===================== 1. ЗАВАНТАЖЕННЯ ДАНИХ =====================
df = pd.read_csv("train.csv")

print("ПЕРШІ 5 РЯДКІВ ДАНИХ:")
print(df.head(), "\n")

# ===================== 2. ВИБІР ОЗНАК =====================
# Ми залишимо тільки кілька числових колонок, щоб було просто
features = ["sex", "has_photo", "has_mobile", "followers_count", "graduation"]
X = df[features]
y = df["result"]

# ===================== 3. РОЗДІЛЕННЯ НА НАВЧАННЯ І ТЕСТ =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===================== 4. СТВОРЕННЯ МОДЕЛІ =====================
model = DecisionTreeClassifier()   # просте дерево рішень
model.fit(X_train, y_train)

# ===================== 5. ПРОГНОЗИ =====================
y_pred = model.predict(X_test)

# ===================== 6. ОЦІНКА =====================
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy*100:.2f}%")

# ===================== 7. ЯСКРАВИЙ ПРИКЛАД =====================
sample = [[1, 0, 1, 100, 1999] ]  
prediction = model.predict(sample)
print("Прогноз для нашого прикладу:", prediction[0])
