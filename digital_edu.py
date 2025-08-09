import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===================== 1. ЗАВАНТАЖЕННЯ ДАНИХ =====================
df = pd.read_csv("train.csv")

print("ПЕРШІ 5 РЯДКІВ ДАНИХ:")
print(df.head(), "\n")

# ===================== 2. ЕТАП EDA (РОЗВІДУВАЛЬНИЙ АНАЛІЗ) =====================
print("ІНФОРМАЦІЯ ПРО ДАНІ:")
print(df.info(), "\n")

print("КІЛЬКІСТЬ ПРОПУСКІВ:")
print(df.isnull().sum(), "\n")

print("ОПИСОВА СТАТИСТИКА ЧИСЛОВИХ ПОЛІВ:")
print(df.describe(), "\n")

print("ГІПОТЕЗИ:")
print("1. Люди з фото частіше купують курс.")
print("2. Чим більше підписників, тим вища ймовірність покупки.")
print("3. Молодші користувачі купують частіше.\n")

plt.hist(df["followers_count"].dropna(), bins=20, color="skyblue")
plt.xlabel("Кількість підписників")
plt.ylabel("Кількість людей")
plt.title("Розподіл підписників")
plt.show()

# ===================== 3. ОЧИЩЕННЯ ДАНИХ =====================
def calc_age(bdate):
    try:
        if len(bdate.split(".")) == 3:
            birth_date = datetime.strptime(bdate, "%d.%m.%Y")
            return datetime.now().year - birth_date.year
    except:
        return None
    return None

df["age"] = df["bdate"].apply(lambda x: calc_age(str(x)))
df["age"].fillna(df["age"].median(), inplace=True)

df["sex"] = df["sex"].map({1: "female", 2: "male"})

df["num_langs"] = df["langs"].apply(lambda x: len(str(x).split(";")))

# Перетворення career_start, career_end у числа
df['career_start'] = pd.to_numeric(df['career_start'], errors='coerce')
df['career_end'] = pd.to_numeric(df['career_end'], errors='coerce')
df['career_start'].fillna(0, inplace=True)
df['career_end'].fillna(0, inplace=True)

# Заповнюємо пропуски в інших числових колонках
df["followers_count"].fillna(0, inplace=True)
df["has_mobile"].fillna(0, inplace=True)
df["has_photo"].fillna(0, inplace=True)

# Видаляємо колонки, які не потрібні
df.drop(columns=["id", "bdate", "last_seen"], inplace=True)

print("ПІСЛЯ ОЧИЩЕННЯ:")
print(df.head(), "\n")

# ===================== 4. ПІДГОТОВКА ДО МОДЕЛІ =====================
# Категоріальні колонки переводимо в числа (one-hot encoding)
df = pd.get_dummies(df, columns=["sex", "education_form", "relation", "education_status", "occupation_type"], drop_first=True)

# Вибір ознак та цільової змінної
X = df.drop("result", axis=1)
y = df["result"]

# ===================== 5. РОЗДІЛ НА TRAIN/TEST =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== 6. НАВЧАННЯ МОДЕЛІ =====================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===================== 7. ОЦІНКА =====================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {acc:.2f}")
